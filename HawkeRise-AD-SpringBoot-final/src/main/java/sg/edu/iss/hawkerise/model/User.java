package sg.edu.iss.hawkerise.model;

import java.util.Collection;

import javax.persistence.ElementCollection;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.OneToMany;

import com.fasterxml.jackson.annotation.JsonIgnore;

@Entity
public class User {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@JsonIgnore
	private int id;
	
	@JsonIgnore
	private String email;

	@ElementCollection
	private Collection<Integer> listHawkers;

	public Collection<Integer> getListHawkers() {
		return listHawkers;
	}

	public void setListHawkers(Collection<Integer> listHawkers) {
		this.listHawkers = listHawkers;
	}

	public User(int id, String email, Collection<Integer>listHawkers) {
		super();
		this.id = id;
		this.email = email;
		this.listHawkers = listHawkers;
	}

	public User(Collection<Integer> listHawkers) {
		super();
		this.listHawkers = listHawkers;
	}

	public User(String email) {
		super();
		this.email = email;
	}

	public User() {
		super();
		// TODO Auto-generated constructor stub
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}


	public String getEmail() {
		return email;
	}

	public void setEmail(String email) {
		this.email = email;
	}

}
