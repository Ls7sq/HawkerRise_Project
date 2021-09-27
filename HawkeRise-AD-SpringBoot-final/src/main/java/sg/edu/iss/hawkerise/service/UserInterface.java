package sg.edu.iss.hawkerise.service;

import sg.edu.iss.hawkerise.model.User;

public interface UserInterface {

	public void update(User user);
	
	public User findUserById(int id);
	
	public void createUser(User user);
	
	public User findUserByEmail(String emial);
}
