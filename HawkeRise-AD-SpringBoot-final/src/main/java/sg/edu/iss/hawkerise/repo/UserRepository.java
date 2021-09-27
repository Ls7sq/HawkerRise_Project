package sg.edu.iss.hawkerise.repo;

import org.springframework.data.jpa.repository.JpaRepository;

import sg.edu.iss.hawkerise.model.User;

public interface UserRepository extends JpaRepository<User, Integer> {
	
	public User findUserById(int id);
	
	public User findUserByEmail(String uname);

}