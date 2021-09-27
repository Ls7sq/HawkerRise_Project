package sg.edu.iss.hawkerise.controller;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;

import javax.servlet.http.HttpSession;
import javax.validation.Valid;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.StringUtils;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import sg.edu.iss.hawkerise.model.Centre;
import sg.edu.iss.hawkerise.model.Hawker;
import sg.edu.iss.hawkerise.model.MenuItem;
import sg.edu.iss.hawkerise.model.Tag;
import sg.edu.iss.hawkerise.service.CentreService;
import sg.edu.iss.hawkerise.service.HawkerService;
import sg.edu.iss.hawkerise.service.TagService;

@Controller
@RequestMapping("/hawker")
public class HawkerController {
	@Autowired
	private HawkerService hservice;

	@Autowired
	private CentreService cservice;

	@Autowired
	private TagService tservice;

	@RequestMapping(value = "/home")
	public String home(@ModelAttribute("hawker") Hawker hawker, Model model, HttpSession session) {
		// if session exists, bring him to the home page directly
		if (session.getAttribute("hsession") == null) {
			return "forward:/hawker/login";
		} else {

			Hawker h = (Hawker) session.getAttribute("hsession");
			model.addAttribute("hawker", h);
			return "hawker/home";
		}
	}

	@RequestMapping(value = "/login")
	public String login(@ModelAttribute("hawker") Hawker hawker, Model model, HttpSession session) {
		// if session exists, bring him to the home page directly
		if (session.getAttribute("hsession") != null) {
			return "forward:/hawker/home";
		} else {
			Hawker h = new Hawker();
			model.addAttribute("hawker", h);
			return "hawker/login";
		}
	}

	@RequestMapping(value = "/logout")
	public String logout(@ModelAttribute("hawker") Hawker hawker, Model model, HttpSession session) {

		if (session.getAttribute("hsession") != null) {
			session.removeAttribute("hsession");
			return "forward:/hawker/authenticate";
		} else {
			return "forward:/hawker/authenticate";
		}
	}

	@RequestMapping(value = "/authenticate")
	public String authenticate(@ModelAttribute("hawker") @Valid Hawker hawker, BindingResult bindingResult, Model model,
			HttpSession session) {

		if (bindingResult.hasErrors()) {
			return "hawker/login";
		}

		if (session.getAttribute("hsession") != null) {
			// if session exists, bring him to the homepage directly
			Hawker h = hservice.findByUserName(hawker.getUserName());
			model.addAttribute("hawker", h);
			return "forward:/hawker/home";
		} else if (hservice.authenticate(hawker)) {
			// if session not exists, check the name and password them bring him to homepage
			Hawker h = hservice.findByUserName(hawker.getUserName());
			session.setAttribute("hsession", h);
			model.addAttribute("hawker", h);
			return "forward:/hawker/home";
		} else
			// if wrong retry
			return "hawker/login";
	}

	@RequestMapping(value = "/register")
	public String register(Model model, HttpSession session) {
		// if session exists, return to the hawker's homepage
		if (session.getAttribute("hsession") != null) {
			session.removeAttribute("hsession");
			return "forward:/hawker/register";
		}
		// if session not exists, bring him to the registration to sign-up
		else {
			Hawker newHawker = new Hawker();
			model.addAttribute("newHawker", newHawker);
			List<Centre> centres = cservice.findAllCentres();
			model.addAttribute("centres", centres);
			List<Tag> tags = tservice.findAllTags();
			model.addAttribute("tags", tags);
			return "hawker/registration";
		}
	}

	@RequestMapping(value = "/completeRegistration")
	public String completeRegisteration(@ModelAttribute("newHawker") @Valid Hawker hawker,
			@RequestParam("filePhoto") MultipartFile multipartFile, Model model, BindingResult bindingResult,
			HttpSession session) throws IOException {
		// if session exists, return to the hawker's homepage
		if (session.getAttribute("hsession") != null) {
			return "forward:/hawker/home";
		}
		// if session does not exist, create the new hawker information
		else {
			List<Centre> centres = cservice.findAllCentres();
			model.addAttribute("centres", centres);
			List<Tag> tags = tservice.findAllTags();
			model.addAttribute("tags", tags);

			if (hservice.checkValidTime(hawker) == false || hservice.checkExists(hawker)) {
				if (hservice.checkValidTime(hawker) == false) {
					bindingResult.addError(new FieldError("hawker", "operatingHours",
							"Closing Hours should be after Opening Hours! Please check."));
				}
				if (hservice.checkExists(hawker)) {
					if (hservice.checkCentreAndUnitNumber(hawker)) {
						bindingResult.addError(
								new FieldError("hawker", "unitNumber", "Unit No. already in use! Please check."));
					}
					if (hservice.checkUserName(hawker)) {
						bindingResult.addError(
								new FieldError("hawker", "userName", "Username already in use! Please check."));
					}
				}
				return "hawker/registration";
			} else {
				Centre belongCentre = cservice.findCentreByName(hawker.getCentre().getName());
				belongCentre.setNumOfStalls(belongCentre.getNumOfStalls() + 1);
				cservice.updateNum(belongCentre);

				if (multipartFile.isEmpty()) {
					ClassPathResource classPathResource = new ClassPathResource("static/img/noPicture.png");
					File originPicture = classPathResource.getFile();
					String fileName = "noPicture.png";
					hawker.setPhoto(fileName);

					hawker.setCentre(belongCentre);

					hservice.createHawker(hawker);

					Hawker savedHawker = hawker;
					String uploadDir = "./hawker-photo/" + savedHawker.getId();

					Path uploadPath = Paths.get(uploadDir);
					if (!Files.exists(uploadPath)) {
						Files.createDirectories(uploadPath);
					}
					try (InputStream inputStream = new FileInputStream(originPicture)) {
						Path filePath = uploadPath.resolve(fileName);
						Files.copy(inputStream, filePath, StandardCopyOption.REPLACE_EXISTING);
					} catch (IOException e) {
						throw new IOException("Could not save uploaded file: " + fileName);
					}

//					String localUrl = "http://10.40.1.56:8080" + hawker.getPhotoImagePath();
					String localUrl = "https://firebasestorage.googleapis.com/v0/b/testing-firebase-64caa.appspot.com/o/images%2F" + fileName + "?alt=media";


					hawker.setHawkerImg(localUrl);

					hservice.createHawker(hawker);

					return "forward:/hawker/login";
				} else {
					String fileName = StringUtils.cleanPath(multipartFile.getOriginalFilename());
					hawker.setPhoto(fileName);

					hawker.setCentre(belongCentre);

					hservice.createHawker(hawker);

					Hawker savedHawker = hawker;
					String uploadDir = "./hawker-photo/" + savedHawker.getId();

					Path uploadPath = Paths.get(uploadDir);
					if (!Files.exists(uploadPath)) {
						Files.createDirectories(uploadPath);
					}
					try (InputStream inputStream = multipartFile.getInputStream()) {
						Path filePath = uploadPath.resolve(fileName);
						Files.copy(inputStream, filePath, StandardCopyOption.REPLACE_EXISTING);
					} catch (IOException e) {
						throw new IOException("Could not save uploaded file: " + fileName);
					}

//					String localUrl = "http://10.40.1.56:8080" + hawker.getPhotoImagePath();
					String localUrl = "https://firebasestorage.googleapis.com/v0/b/testing-firebase-64caa.appspot.com/o/images%2F" + fileName + "?alt=media";

					hawker.setHawkerImg(localUrl);

					hservice.createHawker(hawker);

					return "forward:/hawker/login";
				}

			}
		}
	}

	@RequestMapping(value = "/update")
	public String update(Model model, HttpSession session) {
		if (session.getAttribute("hsession") == null) {
			return "forward:/hawker/login";
		} else {

			Hawker hawker = (Hawker) session.getAttribute("hsession");
			model.addAttribute("hawkerToUpdate", hawker);
			List<Tag> tags = tservice.findAllTags();
			model.addAttribute("tags", tags);
			return "hawker/updateDetails";
		}
	}

	@RequestMapping(value = "/saveUpdate")
	public String saveUpdate(@ModelAttribute("hawkerToUpdate") Hawker hawkerToUpdate, HttpSession session, Model model,
			@RequestParam("filePhoto") MultipartFile multipartFile) throws IOException {
		if (session.getAttribute("hsession") == null) {
			return "forward:/hawker/login";
		}

		if (multipartFile.isEmpty()) {
			Hawker oldHawker = hservice.findByUserName(hawkerToUpdate.getUserName());

			hawkerToUpdate.setPhoto(oldHawker.getPhoto());
			hawkerToUpdate.setHawkerImg(oldHawker.getHawkerImg());

			hservice.update(hawkerToUpdate);

			return "forward:/hawker/home";
		}
	
		
		else {
			String fileName = StringUtils.cleanPath(multipartFile.getOriginalFilename());
			hawkerToUpdate.setPhoto(fileName);

			String uploadDir = "./hawker-photo/" + hawkerToUpdate.getId();

			Path uploadPath = Paths.get(uploadDir);
			if (!Files.exists(uploadPath)) {
				Files.createDirectories(uploadPath);
			}
			try (InputStream inputStream = multipartFile.getInputStream()) {
				Path filePath = uploadPath.resolve(fileName);
				Files.copy(inputStream, filePath, StandardCopyOption.REPLACE_EXISTING);
			} catch (IOException e) {
				throw new IOException("Could not save uploaded file: " + fileName);
			}

//			String localUrl = "http://10.40.1.56:8080" + hawkerToUpdate.getPhotoImagePath();
			String localUrl = "https://firebasestorage.googleapis.com/v0/b/testing-firebase-64caa.appspot.com/o/images%2F" + fileName + "?alt=media";
			hawkerToUpdate.setHawkerImg(localUrl);

			hservice.update(hawkerToUpdate);
			session.removeAttribute("hsession");
			session.setAttribute("hsession", hservice.findByUserName(hawkerToUpdate.getUserName()));

			return "forward:/hawker/home";

		}
		
		
	}

//	@RequestMapping(value = "/saveUpdate")
//	public String saveUpdate(@ModelAttribute("hawkerToUpdate") Hawker hawker, HttpSession session, Model model) {
//		if (session.getAttribute("hsession") == null) {
//			return "forward:/hawker/login";
//		}
//
//		else {
//			hservice.update(hawker);
//			Hawker h = hservice.findByUserName(hawker.getUserName());
//			model.addAttribute("hawker", h);
//			session.removeAttribute("hesssion");
//			session.setAttribute("hsession", h);
//			return "hawker/home";
//		}
//	}
}
