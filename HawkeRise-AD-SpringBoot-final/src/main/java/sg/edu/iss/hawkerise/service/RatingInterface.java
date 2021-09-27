package sg.edu.iss.hawkerise.service;

import sg.edu.iss.hawkerise.model.Rating;

import java.util.List;

public interface RatingInterface {

    public List<Rating> findAllRating();

    public Rating findRatingByStallIdAndEmail(String email, int stallId);

    public void createRating(String email, int stallID, int rating);

    public void updateRating(String email, int stallID, int rating);
}
