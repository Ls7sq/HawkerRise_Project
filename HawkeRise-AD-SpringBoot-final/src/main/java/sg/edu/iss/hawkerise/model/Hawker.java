package sg.edu.iss.hawkerise.model;

import java.util.Objects;
import java.util.Set;

import javax.persistence.CascadeType;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import javax.persistence.Transient;

import com.fasterxml.jackson.annotation.JsonIgnore;

@Entity
public class Hawker {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private int id;
  
	
	@JsonIgnore
	private String firstName;
	@JsonIgnore
	private String lastName;

	private String stallName;
	private String unitNumber;
	public String contactNumber;
	

	@JsonIgnore
	private String userName;
	@JsonIgnore
	private String password;

	@JsonIgnore
	private String[] tags;
	
	private String status;
	private String operatingHours;
	private String closeHours;

	@JsonIgnore
	private String photo;
	private String hawkerImg;
	

	@ManyToOne
	@JoinColumn(name = "centre_id")
	@JsonIgnore
	private Centre centre;

//	@ManyToMany
//	@JoinTable(name = "hawker_tag", joinColumns = @JoinColumn(name = "hawker_id"), inverseJoinColumns = @JoinColumn(name = "tag_id"))
//	private Set<Tag> tags;

	@OneToMany(mappedBy = "hawker", cascade = { CascadeType.REMOVE })
	@JsonIgnore
	private Set<MenuItem> menuItems;

//	@OneToMany(mappedBy = "hawker", cascade = {CascadeType.REMOVE})
//	private Set<Stock> stock;

	public Hawker() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Hawker(int id, String stallName, String unitNumber, String contactNumber, 
			String operatingHours, String closeHours, String hawkerImg, String status) {
		super();
		this.id = id;
		this.stallName = stallName;
		this.unitNumber = unitNumber;
		this.contactNumber = contactNumber;
		this.operatingHours = operatingHours;
		this.hawkerImg = hawkerImg;
		this.status = status;
		this.closeHours = closeHours;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getFirstName() {
		return firstName;
	}

	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public String getLastName() {
		return lastName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public String getStallName() {
		return stallName;
	}

	public void setStallName(String stallName) {
		this.stallName = stallName;
	}

	public String getUnitNumber() {
		return unitNumber;
	}

	public void setUnitNumber(String unitNumber) {
		this.unitNumber = unitNumber;
	}

	public String getContactNumber() {
		return contactNumber;
	}

	public void setContactNumber(String contactNumber) {
		this.contactNumber = contactNumber;
	}

	public String getUserName() {
		return userName;
	}

	public void setUserName(String userName) {
		this.userName = userName;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}

	public String getOperatingHours() {
		return operatingHours;
	}

	public void setOperatingHours(String operatingHours) {
		this.operatingHours = operatingHours;
	}

	public Centre getCentre() {
		return centre;
	}

	public void setCentre(Centre centre) {
		this.centre = centre;
	}

	public Set<MenuItem> getMenuItems() {
		return menuItems;
	}

	public void setMenuItems(Set<MenuItem> menuItems) {
		this.menuItems = menuItems;
	}



	public String[] getTags() {
		return tags;
	}

	public void setTags(String[] tags) {
		this.tags = tags;
	}

	public String getStatus() {
		return status;
	}

	public void setStatus(String status) {
		this.status = status;
	}
	
	public String getPhoto() {
		return photo;
	}

	public void setPhoto(String photo) {
		this.photo = photo;
	}
	
	public String getCloseHours() {
		return closeHours;
	}

	public void setCloseHours(String closeHours) {
		this.closeHours = closeHours;
	}
	
	public String getHawkerImg() {
		return hawkerImg;
	}

	public void setHawkerImg(String hawkerImg) {
		this.hawkerImg = hawkerImg;
	}
	
	@Transient
	@JsonIgnore
    public String getPhotoImagePath() {
        if (photo == null || id <= 0) return null;
         
        return "/hawker-photo/" + id + "/" + photo;
    }
	
	
	@Override
	public int hashCode() {
		return Objects.hash(centre, unitNumber);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Hawker other = (Hawker) obj;
		if ((Objects.equals(centre.getName(), other.centre.getName()) && Objects.equals(unitNumber, other.unitNumber))
				|| Objects.equals(userName, other.userName)) {
			return true;
		} else {
			return false;
		}
	}

}
