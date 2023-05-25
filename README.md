# Linear Regression Model for House Price Prediction

This project implements a linear regression model to predict house prices based on various features.

## Project Structure

The project contains the following files and directories:

- `data/`: Directory containing the dataset file(s).
- `lin_utils.py`: Utility module with functions for training and testing the linear regression model.
- `train.py`: Script to train the linear regression model.
- `test.py`: Script to test the trained model on new data.

## Requirements

- Python 3.x
- pandas 1.2.4
- numpy 1.23.5
- matplotlib 3.5.1

## Usage

1. **Model Training**: Run the `train.py` script to train the linear regression model as well as prepare training data.

    ```
    python train.py
    ```

2. **Model Testing**: After training the model, run the `test.py` script to test the trained model on new data and evaluate its performance.

    ```
    python test.py
    ```

## Data

The project expects the dataset file `data.csv` to be present in the `data/` directory. The dataset should include the following columns:

- `sqft_lot`: Square footage of the lot.
- `sqft_living`: Square footage of the living area.
- `bathrooms`: Number of bathrooms.
- `bedrooms`: Number of bedrooms.
- `condition`: Condition rating of the house.
- `price`: Sale price of the house.

The dataset was orginally sourced from Kaggle and is availible for download [here.](https://www.kaggle.com/datasets/shree1992/housedata)

## Results

After training the model for 20,000 iterations with an initial learning rate of 2.0, and momentum coeffient of 0.9, the model achived a Mean Percent Difference of 29.6% whe compared to true values in the training set and 38.7% for the test set(as of 05/25/2023)
<div>
<img src="https://i.imgur.com/iZeOb75.png" width="467" height="350">
<img src="https://i.imgur.com/sptcKY9.png" width="467" height="350">
<img src="https://i.imgur.com/E59VrmH.png" width="467" height="350">
<img src="https://i.imgur.com/7jTHEmK.png" width="467" height="350">
</div>

## Contributing

Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

