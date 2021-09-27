package sg.edu.iss.hawkerise.repo;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.repository.query.Param;
import sg.edu.iss.hawkerise.model.Rating;

import java.util.List;

public interface RatingRepository extends JpaRepository<Rating, Integer> {

    public Rating findRatingByEmailAndStallID(@Param("email") String email, @Param("stallId") int StallId);

}
