package sg.edu.iss.hawkerise.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import sg.edu.iss.hawkerise.model.User;
import sg.edu.iss.hawkerise.repo.UserRepository;

@Service
public class UserService implements UserInterface{

	@Autowired
	UserRepository urepo;
	
	
	@Override
	public void update(User user) {
		User u = urepo.findUserById(user.getId());
		u.setId(u.getId());
		u.setEmail(u.getEmail());
		u.setListHawkers(u.getListHawkers());
		urepo.saveAndFlush(u);
	}


	@Override
	public User findUserById(int id) {

		return urepo.findUserById(id);
	}


	@Override
	public void createUser(User user) {
		// TODO Auto-generated method stub
		urepo.save(user);
	}


	@Override
	public User findUserByEmail(String email) {
		// TODO Auto-generated method stub
		return urepo.findUserByEmail(email);
	}
}
