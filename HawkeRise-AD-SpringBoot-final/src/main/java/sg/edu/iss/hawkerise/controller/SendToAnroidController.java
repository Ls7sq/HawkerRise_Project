package sg.edu.iss.hawkerise.controller;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import sg.edu.iss.hawkerise.model.*;
import sg.edu.iss.hawkerise.service.*;

@RestController
@RequestMapping(path = "/api")
public class SendToAnroidController {

	@Autowired
	CentreInterface cservice;

	@Autowired
	HawkerInterface hservice;

	@Autowired
	MenuItemInterface mservice;

	@Autowired
	UserInterface uservice;

	@Autowired
	RatingInterface rservice;

	@GetMapping(path = "getAllHawkerCentres")
	public List<Centre> getCentres() {

		List<Centre> resultCentres = new ArrayList<>();
		List<Centre> centreList = cservice.findAllCentres();

		for (Centre centre : centreList) {
			int id = centre.getId();
			String name = centre.getName();
			String address = centre.getAddress();
			String imgUrl = centre.getImgUrl();
			double latitude = centre.getLatitude();
			double longitude = centre.getLongitude();
			int numOfStalls = centre.getNumOfStalls();

			Centre c = new Centre(id, name, address, latitude, longitude, imgUrl, numOfStalls);

			resultCentres.add(c);
		}

		return resultCentres;
	}

	@RequestMapping(path = "getHawkerCentresByDistance/{lat}/{lon}/{distance}")
	public List<Centre> getNearestCentres(@PathVariable("lat") Double lat, @PathVariable("lon") Double lon,
										  @PathVariable("distance") Integer dist) {
		List<Centre> centreList = cservice.findAllCentres();
		List<Centre> nearestCentre = new ArrayList<>();
		for (Centre c : centreList) {
			if (calcDist(lat, lon, c.getLatitude(), c.getLongitude()) < dist) {
				int id = c.getId();
				String name = c.getName();
				String address = c.getAddress();
				String imgUrl = c.getImgUrl();
				double latitude = c.getLatitude();
				double longitude = c.getLongitude();
				double distance = calcDist(lat, lon, c.getLatitude(), c.getLongitude());
				int numOfStalls = c.getNumOfStalls();

				Centre newc = new Centre(id, name, address, latitude, longitude, distance, imgUrl, numOfStalls);

				nearestCentre.add(newc);
			}

		}
		Collections.sort(nearestCentre, new Comparator<Centre>() {
			public int compare(Centre c1, Centre c2) {
				if (c1.getDistance() > c2.getDistance()) {
					return 1;
				}
				if (c1.getDistance() == c2.getDistance()) {
					return 0;
				}
				return -1;
			}
		});

		return nearestCentre;
	}

	@GetMapping(path = "findHawkerStalls/{cid}")
	public List<Hawker> getHawkers(@PathVariable("cid") Integer cid) {

		// get integer (Centre Id) to show the list of hawker belong that Centre.

		List<Hawker> hawkerList = hservice.listHawkers(cid);
		List<Hawker> resultHawkers = new ArrayList<>();

		for (Hawker hawker : hawkerList) {

			int stallId = hawker.getId();
			String stallName = hawker.getStallName();
			String stallUnitNumber = hawker.getUnitNumber();
			String stallContactNumber = hawker.getContactNumber();
			String stallStatus = hawker.getStatus();
			String stallOperatingHours = hawker.getOperatingHours();
			String stallCloseHours = hawker.getCloseHours();
			String stallImg = hawker.getHawkerImg();

			Hawker h = new Hawker(stallId, stallName, stallUnitNumber, stallContactNumber, stallOperatingHours,
					stallCloseHours, stallImg, stallStatus);
			resultHawkers.add(h);
		}

		return resultHawkers;
	}

	@GetMapping(path = "getMenuItems/{id}")
	public List<MenuItem> getMenuItems(@PathVariable("id") Integer id) {

		List<MenuItem> menuItemList = mservice.listMenuItems(id);
		List<MenuItem> resultMenuItems = new ArrayList<>();

		for (MenuItem menuItem : menuItemList) {

			int menuItemId = menuItem.getId();
			String menuItemName = menuItem.getName();
			String menuItemDesc = menuItem.getDescription();
			double menuItemPrice = menuItem.getPrice();
			String menuItemStatus = menuItem.getStatus();
			String menuItemPhoto = menuItem.getLocalUrl();

			MenuItem m = new MenuItem(menuItemId, menuItemName, menuItemDesc, menuItemPrice, menuItemStatus,
					menuItemPhoto);

			resultMenuItems.add(m);
		}

		return resultMenuItems;
	}

	public static float calcDist(double lat1, double lng1, double lat2, double lng2) {
		double earthRadius = 6371000; // meters
		double dLat = Math.toRadians(lat2 - lat1);
		double dLng = Math.toRadians(lng2 - lng1);
		double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(Math.toRadians(lat1))
				* Math.cos(Math.toRadians(lat2)) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
		double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
		float dist = (float) (earthRadius * c);

		return dist / 1000;
	}

	@GetMapping(path = "searchStalls/{reqs}")
	public List<Hawker> searchStalls(@PathVariable("reqs") String reqs) {

		String requirements = reqs.toLowerCase();
		List<Hawker> hawkerList = hservice.listTotalHawkers();
		List<Hawker> resultHawkers = new ArrayList<>();

		for (Hawker hawker : hawkerList) {
			List<MenuItem> menus = mservice.findByHawker(hawker.getUserName());
			String[] taglist = hawker.getTags();
			for (int i = 0; i < taglist.length; i++) {
				taglist[i] = taglist[i].toLowerCase();
			}
			if (hawker.getStallName().toLowerCase().contains(requirements) || menus.stream()
					.filter(x -> x.getName().toLowerCase().contains(requirements)).findAny().isPresent()
					|| Arrays.asList(taglist).contains(requirements)) {
				int stallId = hawker.getId();
				String stallName = hawker.getStallName();
				String stallUnitNumber = hawker.getUnitNumber();
				String stallContactNumber = hawker.getContactNumber();
				String stallStatus = hawker.getStatus();
				String stallOperatingHours = hawker.getOperatingHours();
				String stallCloseHours = hawker.getCloseHours();
				String stallImg = hawker.getHawkerImg();

				Hawker h = new Hawker(stallId, stallName, stallUnitNumber, stallContactNumber, stallOperatingHours,
						stallCloseHours, stallImg, stallStatus);

				resultHawkers.add(h);
			}
		}

		return resultHawkers;
	}

	@GetMapping(path = "getHawkerCentreFromHawkerStall/{hid}")
	public Centre findBelongCentre(@PathVariable("hid") Integer hid) {
		Hawker h = hservice.findById(hid);
		Centre c = h.getCentre();
		return c;
	}

	@GetMapping(path = "favorites/{email}/{hid}")
	public void favorite(@PathVariable("email") String email, @PathVariable("hid") Integer hid) {

		User currentUser = uservice.findUserByEmail(email);
		Hawker hawker = hservice.findById(hid);

		List<Integer> listHawkers = (List<Integer>) currentUser.getListHawkers();
		List<Integer> listResult = new ArrayList<>();

		for (Integer i : listHawkers) {
			if (!listResult.contains(i)) {
				listResult.add(i);
			}
		}

		if (listResult.contains(hawker.getId())) {
			listResult.remove(Integer.valueOf(hawker.getId()));
		} else {
			listResult.add(hawker.getId());
		}

		currentUser.setListHawkers(listResult);
		User u = new User(listResult);
		uservice.update(currentUser);
	}

	// extract user from android and store it in database.
	@GetMapping(path = "saveUser/{email}")
	public void saveUser(@PathVariable("email") String email) {

		User newUser = new User(email);
		uservice.createUser(newUser);

	}


	@GetMapping(path = "listFavourites/{email}")
	public List<Hawker> ListFavorite(@PathVariable("email") String email) {

		User currentUser = uservice.findUserByEmail(email);
		List<Integer> listHawkers = (List<Integer>) currentUser.getListHawkers();
		List<Hawker> hawkerList = hservice.listTotalHawkers();
		List<Hawker> resultHawkers = new ArrayList<>();

		for (Hawker hawker : hawkerList) {

			int hawkerId = hawker.getId();
			if (listHawkers.contains(hawkerId)) {
				int stallId = hawker.getId();
				String stallName = hawker.getStallName();
				String stallUnitNumber = hawker.getUnitNumber();
				String stallContactNumber = hawker.getContactNumber();
				String stallStatus = hawker.getStatus();
				String stallOperatingHours = hawker.getOperatingHours();
				String stallCloseHours = hawker.getCloseHours();
				String stallImg = hawker.getHawkerImg();

				Hawker h = new Hawker(stallId, stallName, stallUnitNumber, stallContactNumber, stallOperatingHours,
						stallCloseHours, stallImg, stallStatus);
				
				resultHawkers.add(h);
			}
		}

		return resultHawkers;
	}
	

	
	@GetMapping(path = "getFavouriteStatus/{email}/{sId}")
	public String getFavouriteList(@PathVariable("email") String email,@PathVariable("sId") int sId)
	{
		//0 means not like
		String likeOrNot = "0";
		  User currentUser = uservice.findUserByEmail(email);
		  List<Integer> listHawkers = (List<Integer>) currentUser.getListHawkers();
		  List<Favourite> favourites = new ArrayList<>();
		  for (Integer i : listHawkers) 
		  {
			  if (i == sId)
			  {
				  likeOrNot = "1";
				  break;
			  }
		  }
		  return likeOrNot;
		  
	}

	@GetMapping(path = "getAllRatings")
	public List<Rating> getAllRatings() {
		return rservice.findAllRating();
	}

	@GetMapping(path = "findRating/{email}/{stallId}")
	public String findRating(@PathVariable("email") String email, @PathVariable("stallId") int stallId) {
		Rating findRating = rservice.findRatingByStallIdAndEmail(email, stallId);
		int theRating;
		if (findRating == null) {
			theRating = 9;
		} else {
			theRating = findRating.getRating();
		}

		return String.valueOf(theRating);
	}

	@GetMapping(path = "setRating/{email}/{stallId}/{rating}")
	public void setRating(@PathVariable("email") String email, @PathVariable("stallId") int stallId,
			@PathVariable("rating") int rating) {
		Rating findRating = rservice.findRatingByStallIdAndEmail(email, stallId);

		if (findRating == null) {
			rservice.createRating(email, stallId, rating);
		} else {
			rservice.updateRating(email, stallId, rating);
		}
	}

}
