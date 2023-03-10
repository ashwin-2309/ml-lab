# #
# Detailed assignment on Feature Engineering.

# Story:

# There is a disease called "Dorona," which is infecting many people in a country. People in the country are very scared due to this disease. The P.M. of the country  Sri. XYZ has announced that he will create a team that will investigate and do an analysis on the disease "Dorona." As the P.M. came to know from some sources, you guys can do efficient data analysis; he is interested in giving this task to you all.

# 1) Data Preparation :

# The data should be prepared using the following statement :

# " The disease "Dorona" increases by 2.4 times every three years."


#                                              D = D_0  * (2.4)**t + Epsilon
# Prepare data starting from 1900 to 2023.
# Add random noise (Epsilon) to it from the gaussian distribution with mean = 0 and S.D = 1
# Make two disjoint sets, "Training set" and "Testing set," with 80% of training data and 20% of testing data.


# 2) Regression :
# Apply linear regression to the collected data.
# Plot the data with the model.
# Show the training and testing accuracy.
# 3) Log transformation + Regression :

# Apply log transformation to the data, then apply regression to the transformed data.
# Plot the data with the model.
# Show the training and testing accuracy.
# 4) Log transformation + Normalization (MinMax) + Regression:

# Apply MinMax normalization to log-transformed data and then apply regression.
# Plot the data with the model.
# Show the training and testing accuracy.
# 5) Log transformation +

# Standardization (Z-score) + Regression:


# Apply standardization on log-transformed data with mean zero and s.d one, and then apply regression.
# Plot the data with the model.
# Show the training and testing accuracy.
