package sg.edu.iss.hawkerise.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import sg.edu.iss.hawkerise.model.Rating;
import sg.edu.iss.hawkerise.repo.RatingRepository;

import java.util.List;

@Service
public class RatingService implements RatingInterface{

    @Autowired
    RatingRepository rRepo;

    @Override
    public List<Rating> findAllRating() {
        return rRepo.findAll();
    }

    @Override
    public Rating findRatingByStallIdAndEmail(String email, int stallId) {
        return rRepo.findRatingByEmailAndStallID(email,stallId);
    }

    @Override
    public void createRating(String email, int stallID, int rating)
    {
        Rating ratingToCreate = new Rating(email, stallID, rating);
        rRepo.saveAndFlush(ratingToCreate);
    }

    @Override
    public void updateRating(String email, int stallID, int rating)
    {
        Rating ratingToUpdate = rRepo.findRatingByEmailAndStallID(email, stallID);
        ratingToUpdate.setRating(rating);
        rRepo.saveAndFlush(ratingToUpdate);
    }
}
