def preprocess_input(data, scaler):
    """Preprocesses input data for prediction."""

    try:
        if 'id' in data.columns:
            data = data.drop(columns=['id'])

        expected_features = scaler.feature_names_in_
        missing_features = set(expected_features) - set(data.columns)

        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Reorder columns (important!)
        data = data[expected_features]

        # Handle missing values (using training data mean)
        data.fillna(data[expected_features].mean(), inplace=True)

        scaled_data = scaler.transform(data)
        return scaled_data

    except (ValueError, KeyError) as e:  # Catch specific errors
        print(f"Preprocessing error: {e}")
        return None  # Return None to indicate failure
