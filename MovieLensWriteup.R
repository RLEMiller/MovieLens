# R script complement to the MovieLensWriteup.Rmd file
#   in satisfaction of the EdX capston project requirements
# ------------ R markdown header -------------------------------------------
#' ---
#' title: "MovieLens Rating Prediction Using SVD and Funk SVD Ensemble Models"
#' author: "Rebecca L.E. Miller"
#' date: "`r format(Sys.time(), '%d %B, %Y')`"
#' output: 
#'   pdf_document: 
#'     keep_tex: yes
#' ---
#' 
## ----setup, include=FALSE, warning=FALSE, message=FALSE, echo=FALSE--------------------------
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("formattable", 
                                     repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", 
                                         repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", 
                                         repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", 
                                         repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", 
                                         repos = "http://cran.us.r-project.org")

library(tinytex)
library(tidyverse)
library(knitr)
library(caret)
library(lubridate)
library(gridExtra)
library(recommenderlab)
library(ggrepel)
library(dplyr, warn.conflicts = FALSE)
options(dplyr.summarise.inform = FALSE) # don't show summarise warnings
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, error = FALSE)


#' --------------------- # Introduction -------------------------------------
# MovieLens file formats in a nice table with knitr::kable
## ----fileFormats, echo = FALSE---------------------------------------------------------------
knitr::kable(data.frame( File = c("ratings.dat", "tags.dat", "movies.dat"),
                         Format = c("UserID::MovieID::Rating::Timestamp",
                                    "UserID::MovieID::Tag::Timestamp",
                                    "MovieID::Title::Genres") ),
             align = c('l','l'),
             format = 'latex',
             linesep = "",
             caption = 'MovieLens 10M Database File Names and Formats')

# Values of genres field from the README for the dataset
## ----GenrePossibles, echo = FALSE------------------------------------------------------------
knitr::kable(list(c("Action","Adventure","Animation","Children's","Comedy"),
                  c("Crime","Documentary","Drama","Fantasy","Film-Noir"),
                  c("Horror","Musical","Mystery","Romance","Sci-Fi"),
                  c("Thriller","War","Western")) ,
             format = 'latex',
             toprule = "",
             linesep = "",
             midrule = "",
             bottomrule = "",
             col.names = "",
             caption = 'MovieLens 10M Genre Options from README file')


#' # Methods
#' 
#' ## Generating Data Sets from the 10M Files and Data Exploration
#' 
#' ### Script Chunk Provided in Course Material
#' 
#' 
## ----ProvidedParser, eval=TRUE, echo=FALSE, warning=FALSE, message=FALSE, cache=TRUE---------

  ##########################################################
  # Create edx set, validation set (final hold-out test set)
  ##########################################################
  
  # Note: this process could take a couple of minutes
  
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
  
  library(tidyverse)
  library(caret)
  library(data.table)
  
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  
  # if using R 3.6 or earlier:
  #movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
  #                                            title = as.character(title),
  #                                            genres = as.character(genres))
  # if using R 4.0 or later:
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                              title = as.character(title),
                                              genres = as.character(genres))
  
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  
  # Validation set will be 10% of MovieLens data
  set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
  test_index <- createDataPartition(y = movielens$rating, times = 1, 
                                    p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
        semi_join(edx, by = "movieId") %>%
        semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)


## ----createTestAndTrain, eval = TRUE, echo=TRUE, cache=TRUE, warning=FALSE, message=FALSE----

    #---------------- Create a Train and Test Set -------------#
    # Create a test set from the edx train set for model development
    test_index <- createDataPartition( y = edx$rating, times = 1, 
                                       p = 0.1, list = FALSE)
    train <- edx[-test_index,]
    temp <- edx[test_index,]
    
    # make sure all userId and movieId in test set are also in train set
    test <- temp %>% 
      semi_join(train, by = "movieId") %>%
      semi_join(train, by = "userId")
    
    # Add rows removed from test set back into train set
    removed <- anti_join(temp, test)
    train <- rbind(train, removed)
    
    # Remove the variables we no longer need
    rm(temp, test_index, edx, removed)


#'   
#' 
#' ### Data Exploration
#' 
#' We examine the data structure and the values in the first few rows.  
#' 
## ----headTrain, eval = TRUE, echo = TRUE-----------------------------------------------------
  head(train,3)


#' #### `userId`  Check out userId structure
## ----userIdcheck, eval=TRUE, echo=TRUE-------------------------------------------------------
  str(train$userId)

#' and the range of values in userId with
## ----echo = TRUE-----------------------------------------------------------------------------
  range(train$userId)

#' and number of distinct users.
## ----echo=TRUE-------------------------------------------------------------------------------
  n_distinct(train$userId)

#' 
#' A histogram of `userId`, 
#' 
## ----VarHistograms, echo=FALSE, warning=FALSE, error=FALSE, fig.cap="(A) Histogram of userId. Level frequency across the range of userId suggests no underlying structure to the variable. (B) Histogram of movieId. The majority of ratings are almost entirely from the first 10,000 movie IDs. (C) Histogram of rating. Shows a higher frequency for natural numbers, or whole-valued ratings, with the majority of ratings in the \"3\" or \"4\" levels. (D) Histogram of timestamp. Shows some potential underlying structure with rating activity varying over time."----
  # a histogram for each variable, plotted in a grid with grid.arrange()
  p1 <- ggplot(train, aes(x = userId)) + 
    geom_histogram(fill = "white", color = "black", bins=30) +
    labs(title = "A Histogram of User IDs")
  p2 <- ggplot(train, aes(x = movieId)) + 
    geom_histogram(fill = "white", color = "black", bins=30) +
    labs(title = "B Histogram of Movie IDs")
  p3 <- ggplot(train, aes(x = rating)) + 
    geom_histogram(fill = "white", color = "black", bins=20) +
    labs(title = "C Histogram of Ratings")
  p4 <- ggplot(train, aes(x = year(as_datetime(timestamp)))) +
    geom_histogram(fill = "white", color = "black", bins=30) +
    labs(title = "D Histogram of Rating Year", x = "timestamp")
  grid.arrange(p1,p2,p3,p4, nrow=2, ncol = 2)
  rm(p1,p2,p3,p4)

#' 
#' #### `movieId`  
#' examine the `movieId` variable type, data range, and the number of distinct movie ID
#' 
## ----movieIdcheck, echo=TRUE-----------------------------------------------------------------
str(train$movieId)

#' And we can see the range of values
## ----echo = TRUE-----------------------------------------------------------------------------
range(train$movieId) 

#' with the number of distinct users found with
## ----echo = TRUE-----------------------------------------------------------------------------
n_distinct(train$movieId) 

#' 
#' 
#' #### `rating`  
#' We now turn to the `rating` variable and find that it is not a continuous variable, but rather can only take on ten values, the half-star values between 0.5 and 5.0, and does not include zero. This suggests that we should consider coercing our predictions to match the range of `rating`, or predicting a "0.5" whenever our rating is less than that value and predicting a "5.0" whenever our prediction is higher than five. We also see that the mean value of `rating` is around 3.5 with a standard deviation of about 1.0, showing that `rating` skews toward more positive values of the available range.
#' Note that 0 is not included in possible rating values
## ----ratingCheck, echo = TRUE----------------------------------------------------------------
  levels( factor(train$rating) )

#' The average rating in the `train` set is found with
## ----echo=TRUE-------------------------------------------------------------------------------
  mu = mean(train$rating)
  mu

#' And the standard deviation with
## --------------------------------------------------------------------------------------------
  sd(train$rating)

#' 
#' A histogram of the `rating` variable reveals that the majority of ratings are either 3's or 4's. The single most probable rating is a 4, but we'll see that the selection of our loss function will drive our choice of the first term in a linear model toward the average rating rather than the most likely rating. Having noted that the actual data is discrete, we will therefore consider coercing our predicted ratings to fall within the range of `rating` and possibly to also coerce it to take on half-rating values, as well.
#' 
#' 
#' #### `timestamp`  
#' Examination of the `timestamp` variable shows that it is, indeed, of the format described in the README file for the 10M database, with a range consistent with the information in that file, as well. We can choose to use the `timestamp` in its provided format or we can convert it to POSIX format with the `lubridate` package. The POSIX format has the advantage of allowing us to use other `lubridate` tools for wrangling date and time objects and is more amenable to human readability.  
#' We see the range of timestamps and the human-readable version with
## ----timestampCheck--------------------------------------------------------------------------
  range(train$timestamp)
  as_datetime(range(train$timestamp))

#' 
#' #### `genres`  
#' 
## ----genresCheck, echo = FALSE, cache=TRUE---------------------------------------------------
  # What are the actual values given in the genres list?
  README_genres_list <- c("Action","Adventure","Animation","Children's","Comedy",
                   "Crime","Documentary","Drama","Fantasy","Film-Noir",
                   "Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
                   "War","Western")

  # sort out a list of all the unique movieIds in edx with the genres for that movie
  movie_genres <- train %>%
    group_by(movieId) %>% 
    summarize(movieId = first(movieId), genres = first(genres))

  # Then get the unique values from those genres
  genres_list <- sort(
    unique(
      unlist(
        strsplit(movie_genres$genres, '\\|'))))

#' 
#' Did the genres list provided in the README file match the dataset?
## --------------------------------------------------------------------------------------------
  identical(genres_list, README_genres_list)

#' The actual values found in the genres variable were
## --------------------------------------------------------------------------------------------
  genres_list

#' Which values are present in our data that weren't in the README?
## --------------------------------------------------------------------------------------------
  setdiff(genres_list, README_genres_list)  

## ---- echo=FALSE-----------------------------------------------------------------------------
  # Clean up
  rm( README_genres_list )

#' 
## ----WranglingFunction, echo=TRUE, warning=FALSE, message=FALSE------------------------------
  WrangleDF <- function(df){
    # extract title & year from the 'title (year)' format, place in separate columns
    df <- df %>% extract(data = . , 
                         col = title, 
                         into = c("title", "year"), 
                         regex = "(.+)[(](\\d+)[)]" )
    # convert timestamp to POSIX format and year to numeric
    df <- df %>% mutate(timestamp = lubridate::as_datetime(timestamp),
                        year = as.numeric(year))
    
    # create a binary column for each genre in our genres_list
    movie_genres <- setNames(
      data.frame( movie_genres$movieId, 
                  lapply(genres_list, function(i) 
                    as.integer( grepl(i, movie_genres$genres))
                   )
      ),
      c("movieId",genres_list) 
    )
    # Join our binarized genres list to the data frame
    df <- df %>% left_join(movie_genres, by = 'movieId')
    # Clean up
    rm(movie_genres)
    return(df)
  }
  
  # Convert the test and train sets
  train <- WrangleDF(train)
  head(train,3)
  
  test <- WrangleDF(test)
  
  # We need to get rid of the "(no genres listed)" title. Those "()" will be 
  # problematic when we want to use the name as a column name in a dataframe.
  train <- train %>% rename( None = "(no genres listed)")
  test <- test %>% rename( None = "(no genres listed)")
  
  # And let's keep the genres_list consistent
  genres_list[genres_list == "(no genres listed)"] = "None"

#' 
## ----genrePrev, echo=FALSE, cache=TRUE, fig.height=3, fig.cap="(A) Prevalence of Genres in train dataset.  (B) Histogram of Release Year"----
  # Calculate the prevalence of each genre in the train set
  genre_prev <- colMeans(train[,None:Western])

  # Visualize prevalence of each genre
  p1 <- data.frame(genre = genres_list, prev = genre_prev * 100) %>% 
    mutate(genre = reorder(genre, prev)) %>% 
    ggplot(aes(genre, prev)) + 
    geom_bar(stat = "identity", fill = "white", color = "black") + 
    coord_flip() +
    labs(y="% of Movies Positive for Genre", x="Genre", title="Prevalence of Genres")
  
  # And the release year
  p2 <- ggplot(train, aes(x = year)) + 
    geom_histogram(fill = "white", color = "black", bins=25) +
    labs(title = "Histogram of Release Year")  
  
  gridExtra::grid.arrange(p1,p2, nrow=1)
  
  # Clean up
  rm( genre_prev, p1, p2 )

#' We see that `genres` is not uniformly distributed 
#' 
#' #### `year`  
## ----yearCheck-------------------------------------------------------------------------------
  range(train$year)


#' ### Data Exploration: Sparsity
## ----sparseUserMovieImage, fig.height=4, fig.cap="Image constructed from one hundred users and movies randomly sampled from the training dataset. There were entire rows and columns with no data because we didn't explicitly select movie and user combinations that were known to have ratings. It is possible to select a subset of users and movies from this datset such that the entire matrix is comprised of NAs."----
  # To use many of the recommenderlab functions, we need to convert our data to a
  # realRatingMatrix format, ie a matrix with items as columns and users as rows

  # Construct a matrix for these users and movies
  # start by setting the seed since we'll use a random sample
  suppressWarnings(set.seed(1234, sample.kind = "Rounding"))
  
  # spread the ratings into a matrix with user rows and movie columns
  y <- train %>% filter( movieId %in% sample( unique(movieId), 100) & 
                           userId %in% sample( unique(userId), 100)) %>%
    select(userId, movieId, rating) %>%
    spread(movieId, rating) %>% 
    as.matrix()
  rownames(y) <- y[,1]  #colnames are already movieId
  y <- y[,-1]   #trim off the userIds column to leave just ratings in matrix
  
  # Use the recommenderlab sparse matrix type with its image function
  recommenderlab::image(as(y,'realRatingMatrix'))
  
  # Clean up
  rm( y )

#' 
#' Users with few ratings
## --------------------------------------------------------------------------------------------
  train %>% group_by(userId) %>% 
    summarize( n = n()) %>% 
    filter(n<=10)

#' Movies with just one rating
## --------------------------------------------------------------------------------------------
  train %>% group_by(movieId) %>% 
    summarize( n = n()) %>% 
    filter(n==1) %>% 
    nrow()

#' 
## ----movieUserCountHist, echo=FALSE, fig.height=3, fig.cap="Histograms of the number of ratings for each movie and user, respectively."----
  # Histogram of number of ratings per user
  p1 <- train %>% group_by(userId) %>% 
    summarize( n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins = 30, fill = "white", color = "black") +
  scale_x_log10() +
  ggtitle("Users")
  
  # Histogram of number of ratings per movie
  p2 <- train %>% group_by(movieId) %>% 
    summarize( n = n()) %>% 
    ggplot(aes(n)) + 
    geom_histogram( bins = 30, fill = "white", color = "black") +
    scale_x_log10() +
    ggtitle("Movies")
  
  # Arrange the two plots together
  gridExtra::grid.arrange(p1,p2, nrow = 1)
  
  # Clean up - these are large plot objects and will eat up RAM
  remove(p1,p2) 

#' 
#' 
#' 
#' ## First Model
## --------------------------------------------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }

## ----justTheMean-----------------------------------------------------------------------------
  # The first model, made obvious by the choice of RMSE as loss function, just mu
  mu <- mean(train$rating)
  
  # create a tibble to store the results
  results <- tibble(method = "Just the average", 
                    RMSE = RMSE(test$rating, mu))
  
  # print a nice table of the results
  kable(results)

#' 
#' 
## ----MovieUserEffects, cache=TRUE------------------------------------------------------------
  # Calculate the mean rating for each user based on the residual of rating
  # less the overall mean rating
  
  # Average rating for each user
  user_mus <- train %>%
    group_by(userId) %>%
    summarize( b_u = mean( rating - mu))
  
  # Create prediction equal to average rating for each user with mu removed
  predicted_ratings <- mu + test %>%
    left_join(user_mus, by = 'userId') %>%
    pull(b_u)
  
  # Add a new result
  results <- results %>% 
    add_row(method = "User Effect Only", 
            RMSE = RMSE(test$rating, predicted_ratings))

  #---------------------- Add Movie Effect ------------------------#
  # The average rating for each movie with mu, user averages removed
  movie_mus <- train %>%
    left_join(user_mus, by = 'userId') %>%
    group_by(movieId) %>%
    summarize(b_i = mean(rating - mu - b_u))
  
  # prediction with user effect then movie effect removed from residual
  predicted_ratings <- test %>%
    left_join(user_mus, by = 'userId') %>%
    left_join(movie_mus, by = 'movieId') %>%
    mutate(prediction = mu + b_i + b_u) %>%
    pull(prediction)
  
  # Add result
  results <- results %>% 
    add_row( method = "User Then Movie Effect",
             RMSE = RMSE(test$rating, predicted_ratings))

  #------- Does it matter if we reverse the order of Movie/User? ----------#
  # Now calculate movie average first
  movie_mus <- train %>%
    group_by(movieId) %>%
    summarize( b_i = mean( rating - mu))

  # Create a prediction based on the average rating for each movie and the 
  # overall average rating for all movies
  movie_effect_predicted_ratings <- mu + test %>%
    left_join(movie_mus, by = 'movieId') %>%
    pull(b_i)
  
  # print out the error
  RMSE(test$rating, movie_effect_predicted_ratings)
  
  # Add to results
  results <- results %>% 
    add_row(method = "Movie Effect Only", 
            RMSE = RMSE(test$rating, movie_effect_predicted_ratings))
  
  # Clean up -  we need to keep our RAM clear
  rm( movie_effect_predicted_ratings)

  #---------------------- Add in User Effect --------------------------#
  # Now re-calculat user effect, but this time with movie mus removed first
  user_mus <- train %>%
    left_join(movie_mus, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_i))
  
  # Calculate prediction with the movie effect, then user effect
  predicted_ratings <- test %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    mutate(prediction = mu + b_i + b_u) %>%
    pull(prediction)
  
  # add row to results
  results <- results %>% 
    add_row( method = "Movie then User Effect",
             RMSE = RMSE(test$rating, predicted_ratings))
  
  # show the nice table of results
  kable(results)


#' 
## ----RegularizationJustificationAnalysis, cache=TRUE-----------------------------------------
  # Do the largest errors in prediction on the test set correlate with having 
  # few ratings in the train set?

  # Number of ratings per user
  n_us <- train %>%
    left_join(user_mus, by = 'userId') %>%
    group_by(userId) %>%
    summarize( n_u = n())

  # Number of ratings per movie
  n_is <- train %>%
    left_join(movie_mus, by = 'movieId') %>%
    group_by(movieId) %>%
    summarize( n_i = n())
  
  # Highest errors with associated samples sizes
  test %>% left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    left_join(n_us, by = 'userId') %>%
    left_join(n_is, by = 'movieId') %>%
    mutate( prediction = mu + b_i + b_u) %>%
    mutate( error = abs(rating - prediction) ) %>%
    arrange(desc(error)) %>%
    slice(1:10) %>%
    select(userId,movieId,rating,n_u,n_i,prediction,error)

#' 
#' Interestingly enough, none of the movies with just one rating appear in our 
#' highest error list. In fact, our results don't really seem to justify the 
#' regularization step that both the "BellKor's Pragmatic Chaos" team and our 
#' coursework took. We can look at our data a bit differently and see how large 
#' the error was among our ratings with the lowest number of ratings per user or 
#' movie.
#' 
## ----errorInLowNs, cache=TRUE----------------------------------------------------------------
  # Smallest sample sizes with associated error
  test %>% left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    left_join(n_us, by = 'userId') %>%
    left_join(n_is, by = 'movieId') %>%
    mutate( prediction = mu + b_i + b_u) %>%
    mutate( error = abs(rating - prediction) ) %>%
    filter( n_i <= 10) %>%
    group_by( n_i ) %>%
    summarize( avg = mean(error))

  # Clean up
  rm( n_is, n_us)

#' 
#' Regularize the movie and user effect to penalize low sample size. We decouple 
#' the two effects by calculating lambdas separately.
## ----regularizationMovie, echo=TRUE, warning=FALSE, message=FALSE----------------------------
  # create sequence of possible lambda values
  lambdas <- seq(0.5,4,0.25)
  
  # loop through the possible values with sapply()
  regularized_rmses <- sapply(lambdas, function(L){
    # recalculate movie averages and predicted ratings as above
    # but instead of the b_i= mean, we now do sum()/(n+L)
    movie_mus <- train %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n() + L))
    
    # prediction as calculated before
    predicted_ratings <- test %>%
      left_join(movie_mus, by = 'movieId') %>%
      mutate(prediction = mu + b_i) %>%
      pull(prediction)
    
    # for each iteration, return the error with that value of L
    return( RMSE(test$rating, predicted_ratings))
  })
  
  # store a plot of the regularization curve
  p1 <- data.frame(lambdas = lambdas, RMSE = regularized_rmses) %>%
    ggplot(aes(lambdas, RMSE)) + 
    geom_point() +
    labs(title="Movie Effect, b_i", y="RMSE", x="lambda")
  
  # print out the minimized error
  min(regularized_rmses)  
  
  # store the movie average lambda that minimized error and show it
  L_i <- lambdas[which.min(regularized_rmses)]  
  L_i

#' 
#' And we can go through the same process to regularize the user effects.  
#' 
## ----regularizationMovieandUser, echo=FALSE, cache=TRUE, message=FALSE, fig.height=3, fig.cap="Regularization curves for movie effect and user effect."----
  # possible lambdas
  lambdas <- seq(2.5,6,0.25)
  
  # loop through
  regularized_rmses <- sapply(lambdas, function(L){
    # re-calculate movie mus with our new value of L_i
    movie_mus <- train %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n() + L_i))
    
    # re-calculate user mus from the new b_i and resulting residual
    user_mus <- train %>%
      left_join(movie_mus, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n() + L) )
    
    # make a prediction using the new b_i and b_u with L_i and L_u
    predicted_ratings <- test %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      mutate(prediction = mu + b_i + b_u) %>%
      pull(prediction)
    
    # return the error
    return( RMSE(test$rating, predicted_ratings))
  })
  
  # make a plot of user average regularization curve and plot with grid.arrange
  p2 <- data.frame(lambdas = lambdas, RMSE = regularized_rmses) %>%
    ggplot(aes(lambdas, regularized_rmses)) + 
    geom_point() +
    labs(title="User Effect, b_u", y="RMSE", x="lambda")
  gridExtra::grid.arrange(p1,p2,nrow=1)
  
  # show the new minimized loss
  min(regularized_rmses)  
  
  # save our minimized lambda
  L_u <- lambdas[which.min(regularized_rmses)]  
  
  #clean up
  rm(p1,p2)

#' Calculate the best estimate after user and movie effect regularization
## ----OverallRegularized, echo=FALSE----------------------------------------------------------
  # Movie average with regularization, b_i
  movie_mus <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + L_i))
  
  # User average with regularization, b_u
  user_mus <- train %>%
    left_join(movie_mus, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + L_u))
  
  # Prediction with regularization
  predicted_ratings <- test %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    mutate(prediction = mu + b_i + b_u) %>%
    pull(prediction)
  
  # add the regularized lowest rmse to results
  results <- results %>% 
    add_row( method = "Movie + User Effect, Regularized",
             RMSE = RMSE(test$rating, predicted_ratings ))
  
  # print nice table
  kable(results)  

#' 
#' Show the ranges of the new, regularized b_i and b_u
## --------------------------------------------------------------------------------------------
  range(movie_mus$b_i)

#' and
## --------------------------------------------------------------------------------------------
  range(user_mus$b_u)

#' when we add the two effects together, we start to get undesirable accumulation of error.
#' 
## --------------------------------------------------------------------------------------------
  # Re-calculate our best prediction so far 
  predicted_residual <- test %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    mutate(prediction = b_i + b_u) %>%
    pull(prediction)
  
  # show range - is outside range of 0.5 to 5.0
  range(predicted_residual)
  
  # clean up
  rm(predicted_residual)

#' 
#' Where the range of the residual should be between 
## --------------------------------------------------------------------------------------------
0.5 - mu

#' and 
## --------------------------------------------------------------------------------------------
5.0 - mu

#' 
#' Create a simple clipping function to coerce our prediction to known range
#' 
## ----SimpleClip------------------------------------------------------------------------------
  SimpleClip <- function(b,UB,LB){
    b[b > UB] <- UB
    b[b < LB] <- LB
    return(b)
  }

#' 
#' And create another function to clip by shrinking/scaling values away from
#' maximas. Sigmoid function:
#' $$
#' G(x) = A + \frac{K-A}{(1 + Qe^{-Bx})^{1/v}}
#' $$
#' With the variables explained in the table 
## ----SigmaFunc-------------------------------------------------------------------------------
  SigmaFunc <- function(x, B = 1, A = 0.5-mu, K = 5-mu, nu = 2){
     Q = (1 - (K/A))^nu - 1
     y <- A + (K - A)/(1 + Q *exp(-B*x))^(1/nu)
     return(y)
  }

#' The sigma variables explained in a table
## ----SigmaVarsExplained, echo=FALSE----------------------------------------------------------
  knitr::kable(data.frame( Effect = c("$A$", "$K$", "$B$",
                                    "$\\nu$", "$Q$"),
                         Description = c("Lower asymptote, default to residual",
                                    "Upper asymptote, default to residual",
                                    "Growth rate, or slope at $x = 0$",
                                    "Tunes slopes near asymptotes, radius of curvature",
                                    "Tunes origin. Found by setting $G(x=0) = 0$ here") ),
                         align = c('l','l'),
                         format = 'latex',
                         linesep = "",
                         caption = 'Variables in Sigmoid Transform',
                         escape=FALSE)

#' 
#' We take a look at the general performance of these two clipping functions below.  
#' 
## ----ClippingFuncsCompared, echo=FALSE, fig.height=4, fig.cap="Comparison of clipping transformation functions. $B = 1$, $\\nu = 1.75$ are good choices for our residual data and should perform in a way comparable to the simple clipping function."----
  # create some generic sequence data and clip it with our two functions
  # then plot the results
  data.frame( xs = seq(-6,5,0.1)) %>%
    mutate( simpleClip = SimpleClip(xs,UB = 2.5, LB = -3),
            sigmaClip1 = SigmaFunc(xs, B=0.1, A = -3, K = 2.5, nu = 2),
            sigmaClip2 = SigmaFunc(xs, B=0.5, A = -3, K = 2.5, nu = 2),
            sigmaClip3 = SigmaFunc(xs, B = 1.0, A = -3, K=2.5, nu = 2)) %>%
    pivot_longer(., cols = simpleClip:sigmaClip3)%>% 
    mutate( name = dplyr::recode(name, 
                          simpleClip = "Simple Clip", 
                          sigmaClip1 = "B = 0.1", 
                          sigmaClip2 = "B = 0.5", 
                          sigmaClip3 = "B = 1.0")) %>%
    mutate(label = ifelse(xs == max(xs), name, NA_character_)) %>%
    ggplot(aes(x = xs, y = value, color = name, group = name)) + 
    geom_line() +
    geom_text_repel(aes(label = label), 
                     nudge_x = 2, na.rm = TRUE) +
    geom_hline(yintercept = -3, linetype = "dashed", color = "red") + 
    geom_hline(yintercept = 2.5, linetype = "dashed", color = "red") +
    theme(legend.position = "none") +
    xlab("Original Value") +
    ylab("Transformed Value") +
    ggtitle("Clipping Methods Mapped to Original Value, Varying B")

#' 
#' Then we can test our estimated error with each of the two clipping functions 
#' 
## ----SimpleClipRMSE--------------------------------------------------------------------------
  # As before, create a sequence of possible values of L
  lambdas <- seq(2.5,6,0.25)
  
  # Re-do regularization with simple clipping
  regularized_rmses <- sapply(lambdas, function(L){
    UB <- 5.0 - mu    # define upper and lower bounds 
    LB <- 0.5 - mu
    
    movie_mus <- train %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n() + L_i))
    
    # clip movie_mus
    movie_mus <- movie_mus %>% mutate(b_i = SimpleClip(b_i, UB, LB))
    
    # calculate user_mus from residula including clipped movie mus
    user_mus <- train %>%
      left_join(movie_mus, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n() + L))
    
    # clip user mus
    user_mus <- user_mus %>% mutate(b_u = SimpleClip(b_u, UB, LB))
  
    # re-define upper and lower bounds for full rating range, not residuals
    UB <- 5.0
    LB <- 0.5
    
    # make prediction from calculated effects
    predicted_ratings <- test %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      mutate(prediction = SimpleClip(mu + b_i + b_u, UB, LB)) %>%
      pull(prediction)
    
    # return the loss
    return( RMSE(test$rating, predicted_ratings))
  })

  # plot the regularization curve with simple clipping and save it
  p1 <- data.frame(lambdas = lambdas, RMSEs = regularized_rmses) %>%
        ggplot(aes(lambdas, RMSEs)) +
        geom_point() +
        labs(title="Simple Clipping")

#' The resulting loss $RMSE$ with the simple clipping procedure is then
## --------------------------------------------------------------------------------------------
min(regularized_rmses) 

#' And the results of our analysis thus far
## ----echo=FALSE------------------------------------------------------------------------------
results <- results %>% 
  add_row( method = "Movie + User Effect, Regularized, Simple Clipping",
           RMSE = min(regularized_rmses))
  
kable(results)

#' With regularization terms
## --------------------------------------------------------------------------------------------
L_u <- lambdas[which.min(regularized_rmses)]  
L_u_simple <- L_u

#' Do the same procedure for sigmoid clipping
#' 
## ----SigmoidalClippingMovieUser, message=FALSE, fig.height=3, fig.cap="Regularization with simple clipping and sigmoid clipping on the user and movie effect model."----
  lambdas <- seq(0,3,0.25)

  regularized_rmses <- sapply(lambdas, function(L){
    UB <- 5.0 - mu
    LB <- 0.5 - mu
    
    movie_mus <- train %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n() + L_i))
    
    # re-scale and coerce movie_mus with the sigmoid transform
    movie_mus <- movie_mus %>% 
      mutate(b_i = SigmaFunc(b_i, A = LB, K = UB))
    
    user_mus <- train %>%
      left_join(movie_mus, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n() + L))
    
    user_mus <- user_mus %>% 
      mutate(b_u = SigmaFunc(b_u, A = LB, K = UB))
  
    UB <- 5.0
    LB <- 0.5
    
    predicted_ratings <- test %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      mutate(prediction = SigmaFunc(mu + b_i + b_u, A = LB, K = UB)) %>%
      pull(prediction)
  
    return( RMSE(test$rating, predicted_ratings))
  })

  # store the regularization curve
  p2 <- data.frame(lambdas=lambdas, RMSEs=regularized_rmses) %>%
          ggplot(aes(lambdas, RMSEs)) +
          geom_point() +
          labs(title="Sigmoid Clipping")
  
  # plot the two clipping regularization curves together and clean up
  gridExtra::grid.arrange(p1,p2,nrow=1)
  rm(p1,p2)


#' Our analysis results up to this point are now:
## ----SigmoidClippingResults, echo=FALSE------------------------------------------------------
  min(regularized_rmses)  
  
  # add clipping results to tibble
  results <- results %>% 
    add_row( method = "Movie + User Effect, Regularized, Sigmoid Clipping",
             RMSE = min(regularized_rmses))
  # print updated table
  kable(results)

#' 
#' Add the resulting b_u and b_i to train and test for convenience in calculating
#' residuals and predictions as we go forward
#' 
## ----AddBuBiTrain, echo=FALSE, message=FALSE-------------------------------------------------
  # Re-calculate movie_mus and user_mus with our trained constants
  # define the upper and lower bounds
  UB <- 5.0 - mu
  LB <- 0.5 - mu
  L_u <- L_u_simple  #L_i didn't change, but L_u did
  
  movie_mus <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + L_i))
  
  # clip b_i
  movie_mus <- movie_mus %>% mutate(b_i = SimpleClip(b_i, UB, LB))
  
  user_mus <- train %>%
    left_join(movie_mus, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = SimpleClip(sum(rating - b_i - mu)/(n() + L_u_simple), UB, LB))
  
  # clip b_u
  user_mus <- user_mus %>% mutate(b_u = SimpleClip(b_u, UB, LB))

  # Create residual column in train set
  train <- train %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    mutate(residual = rating - mu - b_i - b_u) %>%
    mutate(residual = SimpleClip(residual, UB = 5.0-mu, LB = 0.5-mu))

  # Do the same for the test set to save time later
  test <- test %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    mutate( residual = rating - mu - b_i - b_u)

#' 
#' 
#' ## Temporal Effects
#' 
#' Examine the year column data
## ----tempYear, echo=FALSE--------------------------------------------------------------------
  # Plot average rating by release year
  p1 <- train %>% group_by(year) %>%
    summarize(Average = mean(residual), N = n()) %>%
    ggplot(aes(x=year, y=Average)) +
    geom_point(aes(size=N), alpha=0.5, col = I("blue")) +
    geom_line() +
    labs(title="Average Rating by Release Year", x="Year") +
    theme(legend.position = c(0.8,0.8))

#' 
#' ### Age of Movie at Rating Effect
#' 
## ----tempAge, echo=FALSE, fig.height=4, fig.cap="Average rating for movies by release year and by age of movie at the time of rating."----
  # Plot age of movie at rating against average rating for that age
  p2 <- train %>%
    mutate( Age = year(timestamp) - year) %>%
    group_by( Age ) %>%
    summarize(Average = mean(residual), N = n()) %>%
    ggplot(aes(x=Age, y=Average)) +
    geom_point(aes(size=N), alpha=0.5, col = I("blue")) +
    geom_line() +
    labs(title="Average Rating by Age at Rating", x="Year") +
    theme(legend.position = c(0.2,0.8))

  # plot average rating by release year and average rating by age of movie together
  gridExtra::grid.arrange(p1,p2,nrow=1)
  rm(p1,p2)

#' 
#' We see that there is an age effect, though it appears complex.
#' Calculate effect for movie based on time
#'  
## ----b_it, echo=FALSE------------------------------------------------------------------------
  # Find item effects that depend on time. Start with age at time of rating 
  # without considering movies separately. Calculate average rating over all 
  # ratings of that age.
  b_its <- train %>%
    mutate( age = year(as_datetime(timestamp)) - year) %>%
    group_by(age) %>%
    summarize ( b_it = mean(residual))

  # make a prediction
  predicted_ratings <- test %>% mutate(age = year(timestamp) - year) %>%
    left_join(b_its, by = 'age') %>%
    mutate(prediction = mu + b_i + b_u + b_it) %>%
    pull( prediction)
  
  # Add this to results
  results <- results %>% 
    add_row( method = "Add Movie Age at Rating Effect",
             RMSE = RMSE(test$rating, predicted_ratings ))

#' 
## ----b_itClipped, echo=FALSE-----------------------------------------------------------------
  # Find item effects that depend on time. Start with age at time of rating 
  # without considering movies separately. Calculate average rating over all 
  # ratings of that age.
  
  # define the upper and lower bounds
  UB = 5.0 - mu
  LB = 0.5 - mu

  # calculate average rating by age at time of rating and clip to fit in range
  b_its <- train %>%
    mutate( age = year(as_datetime(timestamp)) - year) %>%
    group_by(age) %>%
    summarize ( b_it = mean(residual)) %>%
    mutate(b_it = SimpleClip(b_it, UB = UB, LB = LB))

  # make a prediction based on movie, user, and age effects
  predicted_ratings <- test %>% mutate(age = year(timestamp) - year) %>%
    left_join(b_its, by = 'age') %>%
    mutate(prediction = mu + b_i + b_u + b_it) %>%
    mutate(SimpleClip(prediction, UB = 5, LB = 0.5)) %>%
    pull( prediction)
  
  # Add this to results
  results <- results %>% 
    add_row( method = "Add Movie Age at Rating Effect, Clipped",
             RMSE = RMSE(test$rating, predicted_ratings ))

#' 
#' Above we only group by age. Do the same below, but group by both age and movie.
#'  
## ----PlotAgeByMovieEffect, echo = FALSE, fig.height=4, fig.cap="Rating averages grouped by both movie age and user."----
  # set the seed for reproducibility
  suppressWarnings(set.seed(1234,sample.kind = "Rounding"))
  
  # find the top 100 most rated movies
  movieIds <- train %>% 
    group_by(movieId) %>% 
    summarize(n = n()) %>% 
    top_n(.,100,n) %>% 
    pull(movieId)
  
  # select 12 of those top movies and plot age vs average rating for each of
  # those 12 random movies selected from the top 100 by number of ratings
  train %>% filter( movieId %in% sample(movieIds, 12)) %>% 
    mutate( age = year(as_datetime(timestamp)) - year) %>% 
    group_by(movieId, age) %>% summarize(avg = mean(residual)) %>%
    ggplot(aes(x=age,y=avg, color=as_factor(movieId)) ) + 
    geom_line() + 
    labs(x="Age",y="Average Rating", 
         title="Average Movie Ratings By Age and Movie ID") +
    facet_wrap(vars(movieId))+
    theme(legend.position = "none")

#' 
#' We see that not all movie ratings are dependent on age at the time of rating 
#' in the same way. 
#' 
## ----AgeEffectByMovieAndAge, cache=TRUE, echo=FALSE------------------------------------------
  # Calculate the temporal movie effect - group by movieId and age at rating
  b_its <- train %>%
     mutate( age = year(as_datetime(timestamp)) - year) %>%
     group_by(movieId, age) %>%
     summarize ( b_it = mean(residual))
  
  # make a prediction
  predicted_ratings <- test %>% mutate(age = year(timestamp) - year) %>%
     left_join(b_its, by = c('movieId', 'age')) %>%
     mutate(prediction = mu + b_i + b_u + nafill(b_it, fill=0)) %>%
     pull( prediction)
  
  # add the calculated loss to results tibble
  results <- results %>% 
    add_row( method = "Age Effect by both Age and Movie",
             RMSE = RMSE(test$rating, predicted_ratings ))

#' 
#' If we regularize the movie effect, as grouped by age and user, we get a 
#' regularization curve as seen below.  
#' 
## ----AgeEffectByMovieAndAgeRegularized, echo = FALSE, message=FALSE, warning=FALSE, error=FALSE, fig.height=3, fig.cap="Regularization curve for movie age effect, grouped by user and age of movie at time of rating."----
  # re-load the data.table library, even though we load it at setup
  # ensures this will work when we are working with cached data
  library(data.table)
  
  # Regularize Age Effect by both Age and Movie
  lambdas <- seq(10,50,5)
  
  regularized_rmses <- sapply(lambdas, function(L){
    b_its <- train %>%
      mutate( age = year(timestamp) - year) %>%
      group_by(movieId,age) %>%  # movieId AND age, not just age
      summarize(b_it = sum(residual)/(n() + L))
    
    # make a prediciton with movie effect, user effect, and new movie/age effect
    predicted_ratings <- test %>%
      mutate( age = year(timestamp) - year) %>%
      left_join(b_its, by = c('movieId','age') ) %>%
      mutate(prediction = mu + b_i + b_u + nafill(b_it,fill=0)) %>%
      pull(prediction)
  
    return( RMSE(test$rating, predicted_ratings))
  })
  
  # show the regularization curve
  data.frame(lambdas = lambdas, RMSEs = regularized_rmses) %>%
               ggplot(aes(lambdas, RMSEs)) +
               geom_point() +
               labs(title = "Age Effect, b_it")
  
  # save the regularization constant
  L_it <- lambdas[which.min(regularized_rmses)]
  
  # add to results
  results <- results %>% add_row( 
      method = "Age Effect by Age and Movie, Regularized",
      RMSE = min(regularized_rmses))

#' 
#' When we perform the same regularization task, but also include clipping, our 
#' `RMSE` is
#' 
## ----AgeEffectByMovieAndAgeRegularizedClipped, echo=FALSE------------------------------------
  # Regularize Age Effect by both Age and Movie, Clipped
  # set upper and lower bounds
  UB = 5.0 - mu
  LB = 0.5 - mu
  
  # new possible values of L
  lambdas <- seq(10,50,5)
  
  # loop through L's
  regularized_rmses <- sapply(lambdas, function(L){
    # calculate movie/age effect with sample size penalized
    b_its <- train %>%
      mutate( age = year(timestamp) - year) %>%
      group_by(movieId,age) %>%  # movieId AND age, not just age
      summarize(b_it = sum(residual)/(n() + L)) %>%
      mutate(b_it = SimpleClip(b_it, UB = UB, LB = LB))
  
    # make prediction from movie, user, movie/age effects
    predicted_ratings <- test %>%
      mutate( age = year(timestamp) - year) %>%
      left_join(b_its, by = c('movieId','age') ) %>%
      mutate(prediction = mu + b_i + b_u + nafill(b_it,fill=0)) %>%
      pull(prediction)
  
    return( RMSE(test$rating, predicted_ratings))
  })

  # show new minimum loss and save the regularization constant
  min(regularized_rmses)  
  L_it <- lambdas[which.min(regularized_rmses)]
  
  # add to results
  results <- results %>% add_row( 
      method = "Age Effect by Age and Movie, Regularized, Clipped",
      RMSE = min(regularized_rmses))

#' 
#' And our results up to this point in the analysis are
#' 
## ----b_itToTrain-----------------------------------------------------------------------------
  # We add this temporal effect to the train set and re-calculate the residual
  b_its <- train %>%
      mutate( age = year(timestamp) - year) %>%
      group_by(movieId,age) %>%  # movieId AND age, not just age
      summarize(b_it = sum(residual)/(n() + L_it)) 
  
  # add b_it (movie/age effect) to train, recalc residual
  train <- train %>%
    mutate( age = year(timestamp) - year) %>%
    left_join(b_its, by = c('movieId','age')) %>%
    mutate( residual = rating - mu - b_i - b_u - nafill(b_it, fill=0))
  
  # And do the same with the test set
  test <- test %>%
    mutate( age = year(timestamp) - year) %>%
    left_join(b_its, by = c('movieId','age')) %>%
    mutate( residual = rating - mu - b_i - b_u - nafill(b_it,fill=0))
  
  # show results to this point
  kable(results)

#' 
#' We observe that predicting a rating based on the average age of the movie at 
#' the time of rating, with a different prediction for each movie at each one 
#' year age interval, has a large impact on our loss function. We also note that 
#' regularizing the effect requires a large lambda compared to our other 
#' regularization terms. This suggests that there must be more than forty 
#' ratings for a particular movie when it is a particular age for the average of 
#' those ratings to become predictive. We also note that clipping is either 
#' resulting in no change to our predictions or increasing our error. We will 
#' leave clipping out of the following estimates and re-evaluate its use on the 
#' final model.
#' 
#' 
#' ### Circadian Effect
#' Calculate effect of hour of day for each user on average ratings
#'  
## ----circadianEffect, echo = FALSE, fig.height=4, fig.cap="Average rating by time of day for all users. The total number of ratings at that hour of day is given by the point size. Note that the y axis only covers plus or minus 0.1 rating points. This suggests little predictive power in time of day."----
  # plot overall average rating for all users by hour of day
  # very little predictive power when viewed over all users
  train %>% mutate( hour = hour(timestamp)) %>%
    group_by(hour) %>%
    summarize(avg = mean(residual), N = n()) %>%
    ggplot(aes(hour, avg) ) +
    geom_point( aes(size = N), alpha = 0.6, col = I("blue")) +
    geom_line() +
    ylim(-0.1,0.1) +
    labs(x="Hour of Day",y="Average Rating",
         title="Average Rating by Hour of Day for All Users") +
    theme(legend.position = "bottom")

#' 
#' Look at circadian effect by user
#' 
## ----circadianByUserPlot, echo = FALSE, fig.height=4, fig.cap="Average ratings by hour of day for several prolific raters. Rating patterns are not uniform across users."----
  # Set the seed for random selection of users
  suppressWarnings(set.seed(1234,sample.kind = "Rounding"))
  
  # The top 100 users with most ratings
  userIds <- train %>% group_by(userId) %>% 
    summarize(n = n()) %>% 
    top_n(.,100,n) %>% 
    pull(userId)
  
  # Plot a random sample of 12 top raters by hourly average
  train %>% filter( userId %in% sample(userIds, 12)) %>% 
    mutate( hour = hour(timestamp) ) %>% 
    group_by(userId, hour) %>% 
    summarize(avg = mean(residual)) %>%
    ggplot(aes(x = hour, y = avg, color = as_factor(userId))) + 
    geom_line() + 
    facet_wrap(vars(userId)) +
    theme(legend.position = "none") +
    labs(title="Circadian Rating Patterns for Several Users with Large N", 
         x="Hour", y="Average Rating")


#' We calculate a user effect based on hour of the day using just the average 
#' residual rating given by that user at that hour on any day.  
#' 
## ----circadianByUserb_ut, echo=FALSE---------------------------------------------------------

  # User/time effect or user/hour of day effect, b_ut
  # average rating by user and hour of day
  b_uts <- train %>% mutate( hour =  hour(timestamp)) %>%
    group_by( userId, hour ) %>%
    summarize( b_ut = mean(residual) ) %>%
    select( userId, hour, b_ut)
  
  # Calculate RMSE for the residual predicted by user/hour effect
  predicted_ratings <- test %>%
      mutate( age = year(timestamp) - year, hour = hour(timestamp)) %>%
      #left_join(b_its, by = c('movieId','age')) %>%
      left_join(b_uts, by = c('userId', 'hour')) %>%
      mutate(prediction = mu + b_i + b_u + 
               nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
               nafill(b_ut, fill=0)) %>%
      pull(prediction)
  
  # add to results
  results <- results %>% add_row( 
      method = "Circadian Effect by Hour and User",
      RMSE = RMSE(test$rating, predicted_ratings))  
  
  # nice table of results
  knitr::kable(results)

#' 
#' This was a substantial improvement in RMSE. Does regularization make sense here?
#'  Are there hours of the day for certain users with so few ratings that the 
#'  prediction should be penalized? Our data exploration suggests yes, so we 
#'  should try regularization as seen below.  
#' 
## ----regularizeCircadianByUser, echo = FALSE, fig.height=3, fig.cap="Regularization curve for circadian effect grouped by user."----
  # Regularize user temporal effect - we have done this before so won't comment 
  # extensively going forward
  lambdas <- seq(0,50,5) 
  
  regularized_rmses <- sapply(lambdas, function(L){
    b_uts <- train %>% mutate( hour =  hour(timestamp)) %>%
      group_by( userId, hour ) %>%
      summarize( b_ut = sum(residual)/(n() + L) ) %>%
      select( userId, hour, b_ut)
    
    # Calculate a prediction
    predicted_ratings <- test %>%
      mutate( age = year(timestamp) - year, hour = hour(timestamp)) %>%
      left_join(b_uts, by = c('userId', 'hour')) %>%
      mutate(prediction = mu + b_i + b_u + 
               nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
               nafill(b_ut, fill=0)) %>%
      pull(prediction)
    
      return( RMSE(test$rating, predicted_ratings))
  })
  
  # plot regularization curve for user temporal effect
  data.frame(lambdas = lambdas, RMSEs = regularized_rmses) %>% 
    ggplot(aes(lambdas, RMSEs)) +
    geom_point() +
    labs(title="Regularization Curve for User Temporal Effect")

#' 
#' The regularized $RMSE$ for the circadian effect as grouped by user is
## ----CircadianEffectByUserRMSE---------------------------------------------------------------
  min(regularized_rmses)

#' with training constant, $\lambda_{u,t}$
## ----Lut, cache=TRUE-------------------------------------------------------------------------
  L_ut <- lambdas[which.min(regularized_rmses)]
  L_ut

#' 
## ----echo=FALSE, cache=TRUE------------------------------------------------------------------
  results <- results %>% add_row( 
      method = "Circadian Effect by Hour and User, Regularized",
      RMSE = min(regularized_rmses)) 
  
  knitr::kable(results)

#' 
#' We see that regularization has a large effect, so we re-calculate the 
#' circadian user effect based on the lambda we calculated and add this effect 
#' to the blend.
#' 
## ----addbutTotrain, echo=FALSE---------------------------------------------------------------
  # user hour of day effect, recalculated
  b_uts <- train %>% mutate( hour =  hour(timestamp)) %>%
      group_by( userId, hour ) %>%
      summarize( b_ut = sum(residual)/(n() + L_ut) ) %>%
      select( userId, hour, b_ut)
  
  # add to train set for easier calculations
  train <- train %>%
    mutate(hour = hour(timestamp)) %>%
    left_join(b_uts, by = c('userId', 'hour')) %>%
    mutate(residual = rating - mu - b_i - b_u - 
             nafill(b_it, fill = 0) -   #fill NA with 0, ie no effect
             nafill(b_ut, fill = 0))
  
  # And to test
  test <- test %>%
    mutate(hour = hour(timestamp)) %>%
    left_join(b_uts, by = c('userId', 'hour')) %>%
    mutate(residual = rating - mu - b_i - b_u - 
             nafill(b_it, fill = 0) -   #fill NA with 0, ie no effect
             nafill(b_ut, fill = 0)) 

#' 
#' 
#' The "BellKor" team also found that rating behavior for each user tended to 
#' drift over time. If we look at a random sample of users, as shown below, we 
#' see that there are many users who were not very prolific and for whom a 
#' longer-term drift effect component is not especially useful. If we were to 
#' pursue this strategy, we would have to decide how to treat users without 
#' enough data to establish a trend. One potential strategy would be to combine 
#' all users to consider on overall trend.   
#' 
## ----driftUserTime, warning=FALSE, echo=FALSE, fig.height=4, fig.cap="Drift in user rating over time, by week, for randomly sampled users."----
  set.seed(1234,sample.kind = "Rounding")
  
  # Users with largest sample size
  userIds <- train %>% group_by(userId) %>% 
    summarize(n = n()) %>% 
    top_n(.,100,n) %>% 
    pull(userId)

  # make plot of ratings drift over time for 12 random users in top 100 raters
  train %>% filter( userId %in% sample(userIds, 12)) %>% 
    mutate(week = round_date(timestamp, unit = "week")) %>% 
    group_by(userId, week) %>% 
    summarize(avg = mean(residual)) %>%
    ggplot(aes(week,avg, color = as_factor(userId))) + 
    geom_line() + 
    facet_wrap(vars(userId)) +
    labs(x="Week", y="Average Rating", 
         title="Drift in Ratings Over Time For Several Users With Large Sample Size") +
    theme(legend.position = "none", axis.text.x = element_text(angle=45))

#' 
#' We can observe a slight effect when we combine the data for five hundred 
#' randomly chosen users, as shown below. However, the effect is very slight, 
#' with a slope near zero. Most of the trend slopes for individual users in our 
#' plot are close to zero, as well. Additionally, we have already achieved 
#' approximately the same RMSE as the winning team for the Netflix Prize, and 
#' the computational cost of what we expect to be a very small incremental 
#' improvement begins to offer diminishing returns. We could continue to pursue 
#' these time-based effects on the user side as future refinements.  
#' 
## ----drift500Users, warning=FALSE, echo = FALSE, fig.height=3, fig.cap="Cumulative drift for 500 randomly selected users."----
  # Cumulative drift for 500 random users
  train %>% filter( userId %in% sample(userId, 500)) %>% 
     mutate(week = round_date(timestamp, unit = "week")) %>% 
     group_by(userId, week) %>%
     summarize(avg = mean(residual)) %>%
     ggplot(aes(week,avg)) + 
     geom_point( alpha = 0.2) + 
     geom_smooth(method = lm, formula = y ~ x) +
     labs(x="Week", y="Average Rating", title = "Average Rating Drift Over Time")

#' 
#' 
#' ## Genre Effects
#' 
#' We improved the RMSE between our test and train sets to nearly the same 
#' overall value as the winning team for the Netflix Prize just by considering 
#' user effects that depended on user and a temporal variable and item effects 
#' that depended on item and a temporal variable. We can now begin to consider 
#' how users and items interact. Perhaps the most intuitive next step is to 
#' group movies into item groups based on information already provided to us, 
#' specifically, the `genres` field. We will also attempt to group users and 
#' individual items in a later section, but let's begin with the information 
#' already codified in the dataset.  
#' 
#' ### Predictions with Raw `genres` Data
#' We can look at a random sample of very prolific users and examine how their 
#' average ratings change for the most popular genre groups, where each genre 
#' group is just the raw genres data treated as a factor and converted to a 
#' number. We can see that data represented in the figure below. We see that 
#' there is some similarity between user's rating patterns, but none have 
#' identical patterns. We are only looking at the first ten `genres` and 
#' representing more of the raw genre combinations becomes unreadable very 
#' quickly as there are more than seven hundred. We can reduce the number of 
#' dimensions just by splitting the genres into individual columns and 
#' binarizing the data, as we will do after we examine predictions using just 
#' the raw `genres` field.  
#' 
## ----genresRawAvg, echo=FALSE, warning=FALSE, fig.height = 4, fig.cap="Raw genres as numeric factor vs. the average residual per user for a small random set of prolific users. Some clustering becomes apparent, but there are still far too many variables for this method to be human-readable."----
  suppressWarnings(set.seed(1234,sample.kind = "Rounding"))

  # large sample size genres
  genres_largeN <- train %>% mutate(genres = as_factor(genres)) %>%
    group_by(genres) %>% 
    summarize(n = n()) %>% 
    top_n(.,25,n) %>% 
    pull(genres)
    
  # plot raw genre combination variable against average ratings for 12 randomly
  # selected users from the top 100 raters
  train %>% filter( userId %in% sample(userIds, 12)) %>%
    filter(genres %in% genres_largeN) %>%
    mutate( genres = as_factor(genres)) %>%
    group_by(userId, genres) %>% 
    summarize(avg = mean(residual)) %>%
    ggplot(aes(as.numeric(genres), avg, color = as_factor(userId))) + 
    geom_line( alpha = 0.5) +
    facet_wrap(vars(as_factor(userId))) +
    labs(x="Genres, Numeric Factor Representation", y="Average Rating",
         title="Raw Genres vs Average Ratings for Several Prolific Users") +
    theme(legend.position = "none")

#' 
#' In the following code chunks, we explore how we can improve our prediction 
#' using just the factorized raw genres field, along with regularization. We see 
#' that regularization provides a considerable improvement over the 
#' unregularized version, and this should not be surprising given that `genres` 
#' has more than seven hundred possible values and not all users will have rated 
#' even one movie for every genre, much less enough movies to provide real 
#' predictive power. For users with few ratings overall, there will be multiple 
#' `NA` values, and we can deal with those in a variety of ways. To begin, we 
#' predict those `genres` combinations with no ratings in them as zeroes, 
#' essentially saying that no information contributes no change in the 
#' prediction. We might also consider replacing `NA` values for genres with a 
#' slightly negative value, say -0.1, to indicate that a user who has not rated 
#' any movies in a `genres` category is assumed not to like movies in that 
#' category. This could be a faulty assumption, however, as different users will 
#' have different strategies and reasoning when it comes to which movies they 
#' rate. Some users may choose to provide only negative ratings for movies they 
#' hate, for example, or may like movies in a given genre group reasonably well, 
#' but not well enough to make a concerted effort to rate movies in that genre. 
#' This could be an area for additional refinement.  
#' 
#' We calculate the first user/genre effect estimate in the same way we have 
#' done for previous estimates, but we find that our $RMSE$ has not improved.  
#' 
## ----calculateFirstb_ug, echo=FALSE, error=FALSE, message=FALSE,warning=FALSE----------------
  # user/genre effect as average rating for each user for each genre combination
  b_ugs <- train %>% mutate( genres = as_factor(genres)) %>%
    group_by(userId, genres) %>%
    summarize(b_ug = mean(residual))

  # make a prediction with all effects to this point
  predicted_ratings <- test %>%
      mutate( age = year(timestamp) - year, 
              hour = hour(timestamp),
              genres = as_factor(genres)) %>%
      left_join(b_ugs, by = c('userId', 'genres')) %>%
      mutate(prediction = mu + b_i + b_u + 
               nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
               nafill(b_ut, fill = 0) +
               nafill(b_ug, fill = 0)) %>%
      pull(prediction)
  
  # show loss to this point
  RMSE(test$rating, predicted_ratings)
  
  # add to results
  results <- results %>% add_row( 
    method = "Genre Effect, just Factors",
    RMSE = RMSE(test$rating, predicted_ratings)) 

#' 
#' We try regularizing as we have previously observed that there are many genre 
#' combinations for which users have given no ratings. We see that regularization 
#' does, indeed, help quite a bit in this case as seen in the regularization 
#' curve below.  
#' 
## ----regularizeGenreEffect, fig.height=3, fig.cap="Regularization curve for genre effect based on factorized original genres string."----
  # regularize user/genre effect to penalize small sample size
  # possible values of L
  lambdas <- seq(7,11,0.5)
  
  # loop through
  regularized_rmses <- sapply(lambdas, function(L){
    b_ugs <- train %>% mutate( genres = as_factor(genres)) %>%
      group_by(userId, genres) %>%
      summarize(b_ug = sum(residual)/(n() + L))

    predicted_ratings <- test %>%
      mutate( age = year(timestamp) - year, 
              hour = hour(timestamp),
              genres = as_factor(genres)) %>%
      left_join(b_ugs, by = c('userId', 'genres')) %>%
      mutate(prediction = mu + b_i + b_u + 
               nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
               nafill(b_ut, fill = 0) +
               nafill(b_ug, fill = 0)) %>%
      pull(prediction)
    return(RMSE(test$rating, predicted_ratings))
  })

  # plot regularization curve
  data.frame(lambdas = lambdas, RMSEs = regularized_rmses) %>%
    ggplot(aes(lambdas, RMSEs)) +
    geom_point() +
    labs(title = "Regularization of Genre Effect, b_ug")

#' 
#' Our regularized minimum $RMSE$ was
## --------------------------------------------------------------------------------------------
  min(regularized_rmses)

#' with regularization parameter
## --------------------------------------------------------------------------------------------
  L_ug <- lambdas[which.min(regularized_rmses)] 

## ----echo = FALSE----------------------------------------------------------------------------
  # add to the results
  results <- results %>% add_row( 
    method = "Genre Effect, just Factors, Regularized",
    RMSE = min(regularized_rmses)) 

#' We will explore another method for estimating the genre effect before we 
#' commit the predictions to our blend.  
#' 
#' ### Predictions with Binarized `genres`
#' We created the binarized columns for `genres` in our Data Wrangling section. 
#' We can now begin to explore that data by looking at how the residual differs 
#' for each of the users in our randomly selected small group. We see in the 
#' following figure that for genres with many ratings by that user, the range 
#' within each genre is large, so even if we could predict the residual with 
#' just one genre, our error would still be large. The problem is that it 
#' doesn't really make sense to predict the residual with one genre when many of 
#' the items, or movies, in our dataset belong to more than one genre group, so 
#' the residual depends on multiple genre effects. That is why the model proposed 
#' in the coursework was a linear combination of the effects of the various 
#' genres. The big-data problem of trying to do a linear regression on millions 
#' of data points is still restrictive, just as it would be restrictive, in 
#' terms of time and computational resources, to do a direct linear regression 
#' for `rating` predicted by `userId` and `movieId`. One possible approach is to 
#' reduce the number of dimensions. We can see in our data that there is 
#' substantial correlation between genres for each user. For example, `Action` 
#' and `Adventure` have similar means and distributions for each of the `userIds` 
#' even though the distributions differ between users. So we see that we could 
#' both reduce the number of dimensions and then perhaps group users together, 
#' thereby reducing the size of a user/item matrix in both dimensions. A 
#' possible approach would be to do Principal Component Analysis to reduce the 
#' number of genres, followed by k-Means Clustering to reduce the number of 
#' users into user groups and finally to do a linear model predicting residual 
#' based on principal components of genre and user clusters.  
#' 
## ----residualByGenreFiveUsers, echo=FALSE, warning=FALSE, message=FALSE, fig.height=5, fig.cap="Average residual rating for each genre for five randomly selected users."----
  # set seed for reproducibility
  set.seed(1234,sample.kind = "Rounding")
  
  # select five random users
  users <- sample(train$userId, 5)
  
  # plot binarized (separated) genres for those users
  train %>% filter(userId %in% users) %>% 
    select(userId, residual, None:Western) %>%
    pivot_longer(., cols = None:Western, names_to = 'genre', 
                 values_to = 'yes') %>%
    filter( yes == 1) %>%
    ggplot( aes( x = as_factor(genre), 
                 y = residual, 
                 color = as_factor(userId))) +
      geom_boxplot(coef=3) + 
      geom_jitter(width = 0.25, alpha = 0.5) + # spread out points horizontally
      facet_wrap( ~userId, ncol = 1) + # split out users into different subplots
      labs(x="Genre", y="Average Residual Rating", 
           title = "Binarized Residual Ratings for Five Random Users") +
      theme(legend.position = "none", 
            axis.text.x = element_text(angle = 45, 
                                       hjust = 1))
  
  remove(users)

#' 
#' 
## ----GenreModelAverages----------------------------------------------------------------------
  # Before doing anything complicated, how does an averaging approach improve 
  # RMSE? For each user/genres set, calculate the average residual. Generate a 
  # prediction by averaging over all applicable residuals for the user/genres 
  # in test.
  dummy <- train[,None:Western] * train$residual
  
  # Convert 0 to NA so we can easily remove those rows that don't apply
  # to the genre in question
  dummy[dummy==0] <- NA
  dummy <- dummy %>% mutate(userId = train$userId)
  
  # use dplyr::across to compute column means, removing NAs - 
  # this computes the user's mean residual rating for each genre
  genre_avgs <- dummy %>% group_by(userId) %>% 
    summarize(dplyr::across(.cols = None:Western, 
                            .fns = ~ mean(.x, na.rm = TRUE)))
  
  # Now estimate a residual for the test set based on userId and the genres
  # of the movies rated
  # Our best previous prediction for the ratings in test was:
  predicted_ratings <- test %>%
    mutate(prediction = mu + b_i + b_u + 
             nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
             nafill(b_ut, fill=0)) %>%
    pull(prediction)
  
  #This gives unexpected result in terms of shape of regularized curve
  prediction_genres <- test %>%
    left_join(., genre_avgs, by = 'userId', suffix = c("","_avg")) %>%
    mutate( pred = rowMeans(as.matrix(select(.,None:Western)) *
                    as.matrix(select(.,None_avg:Western_avg)),
                    na.rm = TRUE
                    )) %>%
    pull(pred)

  
  # What is the RMSE for our best prediction plus the binarized genre 
  # effect?
  results <- results %>% add_row( 
    method = "Genre Effect, Binarized",
    RMSE = RMSE(test$rating, predicted_ratings + prediction_genres))  

#'
#' We see that binarizing the `genres` data does not improve our loss function.
#' We should attempt to regularize this model, as well, because we saw that
#' there were many `userId` and individual genre combinations with either few
#' data points or no data.
#'
#' We noted that some users have not rated any movies in a particular genre, and
#' very few in others, therefore it might make sense to penalize estimates of
#' these, or regularized on sample size as we have done previously as seen in
#' the following figure. For future work, it might also make sense to try
#' filling NAs with something other than 0, perhaps -0.5 or -1 on the assumption
#' that if the user has not rated movies in that genre, they don't like that
#' genre.
#' 
## ----GenreRegularization, echo=FALSE, warning=FALSE, message=FALSE, cache=TRUE, fig.height=3, fig.cap="Binarized Genre Regularization"----
  lambdas <- seq(0,20,5)
  regularized_rmses <- sapply(lambdas, function(L){
    # re-calculate the averages over genre
    genre_avgs <- dummy %>% group_by(userId) %>% 
     summarize(dplyr::across(.cols = None:Western,
                             .fns = ~ sum(.x, na.rm = TRUE)/
                                     ( sum(!is.na(.x)) + L)
                            )
              )
  
    
    # Now estimate a residual for the test set
    prediction_genres <- test %>% 
      left_join(., genre_avgs, by = 'userId', suffix = c("","_avg")) %>%
      mutate( pred = rowMeans(as.matrix(select(.,None:Western)) *
                      as.matrix(select(. ,None_avg:Western_avg)),
                      na.rm = TRUE
                      )) %>%
      pull(pred)
  
    # What is the RMSE for our best prediction plus the genre effect?
    return(RMSE(test$rating, predicted_ratings + prediction_genres))
  })

  # plot regularization curve
  data.frame(lambdas = lambdas, RMSEs = regularized_rmses) %>% 
    ggplot(aes(lambdas, RMSEs)) +
    geom_point() +
    labs(title="Binarized Genre Effect, b_ug")

#' Our resultant $RMSE$ was
## ---- echo=FALSE-----------------------------------------------------------------------------
  min(regularized_rmses)

#' 
## ----echo=FALSE------------------------------------------------------------------------------
  # add binarized result to tibble
  results <- results %>% add_row( 
    method = "Genre Effect, Binarized, Regularized",
    RMSE = min(regularized_rmses)) 
  
  # nice table of results
  kable(results)
  
  # Do some cleanup
  rm( dummy, prediction_genres, predicted_ratings, lambdas, regularized_rmses )

#' 
#' We didn't improve our loss function with the binarization approach, but we
#' might be able to use the format to create a decision tree or k nearest
#' neighbor model of residual predicted by genres. We examine the heatmap of
#' user vs. genre in the following figure.
#' 
## ----Genres, message=FALSE, warning=FALSE, fig.cap="Heatmap of binarized genres average rating for several randomly chosen users. Binarized genres with near zero variance are removed and users and genres are clustered by distance. Red color indicates a negative residual and blue indicates a positive residual."----
  # We can begin by asking which genres are not predictive, or have very little variance over the range of our data
  nzv <- nearZeroVar( 
    sweep(train %>% select(None:Western), 1, train$residual, "*") )

  # These correspond with the following genres
  genres_list[nzv]
  
  # If we select a small, random group of users, we can see if filling in
  # the missing values with one of the recommenderlab functions will 
  # help to improve our loss function
  
  # We start with a small version using a random set of users - set seed
  suppressWarnings(set.seed(1234,sample.kind = "Rounding"))
  
  # Randomly sample 100 users from the training set
  users <- sample(train$userId, 100)
  
  # create matrix of userId rows and average rating for each genre in cols
  x <- genre_avgs %>% 
    filter( userId %in% users) %>%
    as.matrix() 
  
  rownames(x) <- x[,1] #create rownames
  x <- x[,-1]  #trim off userIds to leave just residuals
  x[is.na(x)] <- 0  # heatmap has an na.rm = T option, but still errs
  
  # make a heatmap of the genres with some variance vs users
  # this function shows the distance-based cluster on the margins
  heatmap(x, col = RColorBrewer::brewer.pal(11,"RdBu"), na.rm = TRUE)
  
  # Clean up
  rm( nzv, genre_avgs )


#'  
#' We see that some relationships between groups of users and genres do exist.
#' For example, at the lower right we see a group of users defined by a strong
#' dislike of `Film-Noir`. Just above that group we have one that dislikes
#' `Children` and `Animation` genres. User 71055 belongs to a group that likes
#' `Documentary` and `Film-Noir` but mildly dislikes `Action`. It appears that
#' the `Children` and `Animation` columns are positively correlated just based
#' on a similarity of residual patterns.
#'
#' Therefore perhaps a neighborhood or clustering approach would improve our
#' model. If we try the following code to fit a k-Nearest Neighbor model, we
#' find that we run into problems with too many ties in our data. This is due to
#' the fact that we have integer, and worse, binary data, for all of our
#' predictors. We can deal with this by either increasing the size of the
#' neighborhood, or by adding very small random noise to our binary matrix data.
#' We could also edit the source code for knn3, which throws the error for too
#' many ties. We should consider this very carefully, however, to determine if
#' it is appropriate or worthwhile. Another option is to fit the k-Nearest
#' Neighbor model with the original `genres` data, but treat the list of genres
#' as factors.
#' 
## ----GenreKnnDummy, eval = FALSE-------------------------------------------------------------
##   train_small <- train %>% filter(userId %in% users) %>%
##     mutate(userId = as.factor(userId), genres = as.factor(genres)) %>%
##     select(userId, genres, residual)
##   test_small <- test %>% filter(userId %in% users) %>%
##     mutate(userId = as.factor(userId), genres = as.factor(genres)) %>%
##     select(userId, genres, residual)
## 
##   # Note that just doing knn3(residual ~ ., data = train_small, K=100)
##   # results in an error for too many ties
##   knn_fit <- knn3(residual ~ ., data = train_small, k = 5)
##   y_hat <- predict(knn_fit, test_small, type = "class")
##   RMSE(test$residual, y_hat)
## 

#' 
#' The above code is not evaluated because it results in an error. The new data in `test_small` contains levels that were not present in `train_small` set. We could go through the process of adding the missing levels as rows with `NA` or zero values for the `residual`, but this amounts to adding rows, and in fact data, to our original dataset. We must also consider that time spent wrangling data to allow us to fit a particular model is part of the overall return of the model. That is to say, we need to balance improved RMSE with compute time and development time. So rather than continuing to develop a k-Nearest Neighbor model, or other distance-based  model such as Random Forest, we can explore the utility of matrix factorization methods for this application. Below, we introduce the RecommenderLab tools for this purpose.
#' 
#' ### Predicting Genre Interaction with Matrix Factorization
#' 
## ----RecommenderGenres, cache=TRUE, message=FALSE, error=FALSE-------------------------------
  # Create an average ratings matrix for genres (original data) and users
  # similar to above
  x <- b_ugs %>% filter(userId %in% users) %>%
    spread(genres, b_ug) %>% 
    as.matrix()
  rownames(x) <- x[ ,1]  # add row names
  x <- x[ ,-1]  # remove the userId column to leave just ratings in matrix

  # We will want to try multiple methods from the recommenderlab, so
  # create a function to handle the repetitive work
  GenresCalcRecommenderRMSE <- function(x, Method){
    # Fit the model
    rec <- Recommender( as(x,'realRatingMatrix'), method = Method)
    
    # Make a prediction to fill in missing data
    predictions <- as( predict(rec, as(x,'realRatingMatrix'), 
                                 type = "ratingMatrix"), 
                         "data.frame") %>%
      mutate(userId = as.numeric(.$user),  #format generic dataframe names
             genres = item , 
             residual_hat = rating) %>% 
      select(userId, genres, residual_hat)
    
    # join the test data with the prediction data frame
    SVDrecs <- test %>% select(userId, genres, residual) %>%
      filter( userId %in% users) %>%
      left_join(., predictions, by = c('userId', 'genres')) %>%
      filter(!is.na(residual_hat))
    
    # And calculate the loss function, RMSE
    RMSE(SVDrecs$residual, SVDrecs$residual_hat)
  }
  
  
  # The available methods for real rating matrix types
  methods = c('SVD','SVDF','UBCF','IBCF','ALS')
  
  # See which method performs best on our small dataset
  recommenderRMSEs <- sapply(methods, function(method)
    GenresCalcRecommenderRMSE(x, method))
  
  # show resulting loss function values
  recommenderRMSEs


#' 
#' We can perform the same task as above, approximately, with `recommenderlab`
#' evaluation tools. We create a larger dataset initially because the evaluation
#' scheme controls test/train proportion and how many times we do a test.
#' 
## ----RecommenderGenresRL---------------------------------------------------------------------

  # Create an average ratings matrix for genres (original data) and users
  suppressWarnings(set.seed(1234,sample.kind = "Rounding"))
  
  # Randomly sample 100 users from the training set
  users <- sample(train$userId, 1000)

  # Build the realRatingMatrix of user vs genres filled with residual ratings
  x <- b_ugs %>% filter(userId %in% users) %>%
    spread(genres, b_ug) %>% 
    as.matrix()
  rownames(x) <- x[ ,1]
  x <- x[ ,-1]
  x <- as(x, 'realRatingMatrix')  # convert x matrix to recommenderlab sparse type
  
  # Define a recommenderlab rating scheme
  scheme <- evaluationScheme( x, 
                              method = "split", # use cross-validation
                              train = 0.9, # 90 percent train
                              k = 2, # just 2 components to start
                              given = -5) # all but 5, use 5/user as test set

  # algorithms to train with parameters
  algorithms <- list(
    "Random" = list(name = "RANDOM", param = NULL),
    "Popular" = list(name = "POPULAR", param = NULL),
    "ALS Latent Factors" = list(name = "ALS", 
                                param = list(n_iterations = 5)),
    "User-Based CF" = list(name = "UBCF",param = list(nn=25)),
    "Item-Based CF" = list(name = "IBCF", param = list(k=30)),
    "SVD" = list(name = "SVD", param = list(k=10)),
    "Funk SVD" = list(name = "SVDF", 
                      param = list(min_epochs = 10, max_epochs = 50))
  )

  # Evaluate the algorithms according to the scheme, use ratings
  recommenderGenreResults <- evaluate(scheme, algorithms, type = "ratings") 
  
  # Do some cleanup
  rm( algorithms, recommenderGenreResults, scheme )

#'
#' We see that Funk SVD performed best, followed by ALS. Unfortunately, these
#' were also the two most time-consuming methods. Part of the reason for this is
#' that both SVDF and ALS allow for more control than just SVD, for example. For
#' this reason, we will optimize the parameters with our smaller training set of
#' randomly selected users.
#'
#'
#' In the code below, we tune the Funk SVD algorithm model of genre preference
#' for each user. There are several parameters that we could tune with the
#' `recommenderlab` SVDF method. These include
#' 
## ----SVDFtrainingTerms, echo = FALSE, cache=TRUE---------------------------------------------
knitr::kable(data.frame( Effect = c("k", "$\\gamma$", "$\\lambda$",
                                    "min epochs", "max epochs"),
                         Description = c("rank of the matrix/number of features/number of principal components",
                                    "regularization term that penalized large values",
                                    "learning rate",
                                    "min number of iterations per feature",
                                    "max number of iterations per feature") ),
             align = c('l','l'),
             format = 'latex',
             linesep = "",
             caption = 'Training terms for SVDF in recommenderlab',
             escape=FALSE)

#' 
#' 
## ----GenresSVDF, cache=TRUE------------------------------------------------------------------
  # Tune the parameters for the genres Funk SVD matrix factorization
  tuneGenreSVD <- sapply(c(2,7,12), function(K)
    sapply(c(0.01, 0.025, 0.05), function(Gamma){
      # Fit the model
      rec <- Recommender( x, method = 'SVDF', 
                          list(k = K, 
                               gamma = Gamma,
                               normalize = NULL))
      
      # Make a prediction to fill in missing data. User recommenderlab
      # predict function and output as data frame
      predictions <- 
        as( predict(rec, as(x,'realRatingMatrix'), type = "ratingMatrix"), 
                           "data.frame") %>%
        mutate(userId = as.numeric(.$user),  #format generic dataframe names
               genres = item , 
               residual_hat = rating) %>% 
        select(userId, genres, residual_hat)
      
      # join the test data with the prediction data frame
      SVDrecs <- test %>% select(userId, genres, residual) %>%
        left_join(., predictions, by = c('userId', 'genres')) %>%
        filter(!is.na(residual_hat))
    
      # And calculate the loss function, RMSE
      return(RMSE(SVDrecs$residual, SVDrecs$residual_hat))
  }))
  
  # Make the results readable
  colnames(tuneGenreSVD) <- c("k = 2", "7", "12")
  rownames(tuneGenreSVD) <- c("G = 0.01", "0.025", "0.05")
  kable(tuneGenreSVD)

#' 
## ----AssignSVDFtrainingterms, cache=TRUE-----------------------------------------------------
  # Assign our trained k and gamma to use in final model
  kGenreSVDF <- 12
  GammaGenreSVDF <- 0.025
  
  #Clean up
  rm(tuneGenreSVD)

#'
#' We see that our best results were achieved with the largest number of
#' features trained, a number higher than the default of ten. Our best
#' regularization was achieved with a value just slightly above the one
#' mentioned in the Netflix Challenge Update by Simon Funk that is referenced in
#' the `recommenderlab` manual
#' \footnote{https://sifter.org/~simon/journal/20061211.html} as well as just
#' slightly above the default value. With this somewhat minimal training, we can
#' calculate our final genre effect based on the entire set of user and genre
#' pairs.
#' 
## ----AddGenreEffectToTestTrainResidual, eval = TRUE------------------------------------------
  # Re-calculate the user/genre interactions with the new k = 12, gamma = 0.025
  
  # Calculate new matrix for full set of user/genre pairs  
  x <- b_ugs %>%
    spread(genres, b_ug) %>% 
    as.matrix()
  rownames(x) <- x[ ,1]
  x <- x[ ,-1]
  x <- as(x, 'realRatingMatrix')  

  # Fit the model
  rec <- Recommender( x, method = 'SVDF', 
                      list(k = kGenreSVDF, 
                           gamma = GammaGenreSVDF,
                           verbose = FALSE,
                           normalize = NULL))
  
  # Make a prediction to fill in missing user/genre interactions
  b_ugs <- as( predict(rec, x, type = "ratingMatrix"), "data.frame") %>%
    mutate(userId = as.numeric(.$user),  #format generic dataframe names
           genres = item , 
           b_ug = rating) %>% 
    select(userId, genres, b_ug)

  # Add prediction elements to train set
  train <- train %>%
      mutate( genres = as_factor(genres)) %>%
      left_join(b_ugs, by = c('userId', 'genres')) %>%
      mutate(residual = rating - b_i - b_u - 
               nafill(b_it, fill = 0) -   #fill NA with 0, ie no effect
               nafill(b_ut, fill = 0) -
               nafill(b_ug, fill = 0))
  
  # Add prediction elements to test set
  test <- test %>%
    mutate( genres = as_factor(genres)) %>%
    left_join(b_ugs, by = c('userId', 'genres')) %>%
    mutate(residual = rating - b_i - b_u - 
               nafill(b_it, fill = 0) -   #fill NA with 0, ie no effect
               nafill(b_ut, fill = 0) -
               nafill(b_ug, fill = 0))
  
  # calculate prediction
  predicted_ratings <- test %>%
    mutate(prediction = mu + b_i + b_u + 
             nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
             nafill(b_ut, fill=0) +
             nafill(b_ug, fill = 0)) %>%
    pull(prediction)
  
  # show loss result
  RMSE(test$rating, predicted_ratings )  

#' 
## ----echo=FALSE,cache=TRUE,warning=FALSE, error=FALSE----------------------------------------
  # add the result to the tibble  
  results <- results %>% 
      add_row(method = "Genre, Just Factors, Regularized, SVDF", 
              RMSE = RMSE(test$rating, predicted_ratings))

  # Try clipping the result to fit in known range
  predicted_ratings <- SimpleClip(predicted_ratings, UB = 5.0, LB = 0.5)

  # Calculate the loss
  RMSE(test$rating, predicted_ratings ) 
  
  # Add to results
  results <- results %>% 
    add_row(method = "Genre, Just Factors, Regularized, SVDF, clipped", 
            RMSE = RMSE(test$rating, predicted_ratings))

#' 
#' 
#' ## Movie User Interaction Effect Having estimated many different baseline
#' effects, and the user/genre effect, which is related to the user/movie effect
#' but contains less information, we finally get to the core of the machine
#' learning task presented to us in this capstone assignment. We will use
#' `recommenderlab` to examine movie/user interactions. To start, we'll need to
#' create a ratings matrix from the residuals. We start with a small version
#' using a random set of users.
#' 
## ----MovieUserInteractions, cache=TRUE, warning=FALSE, message=FALSE-------------------------

  # set the seed for reproducibility
  suppressWarnings(set.seed(1234,sample.kind = "Rounding"))
  
  # Randomly sample 1000 users from the training set
  users <- sample(train$userId, 1000)
  
  # Construct ratings matrix for training recommender models
  x <- train %>% select(userId, movieId, residual) %>%
    filter( userId %in% users ) %>%
    spread(movieId, residual) %>%
    as.matrix() 
  rownames(x) <- x[,1] #create rownames
  x <- x[,-1]  #trim off userIds to leave just residuals
  x <- as(x, 'realRatingMatrix') # make sparse matrix

  # We will want to try multiple methods from the recommenderlab, so
  # create a function to handle the repetitive work
  CalcRecommenderRMSE <- function(x, Method){
    # Fit the model
    rec <- Recommender( x, method = Method)
    
    # Make a prediction to fill in missing data
    predictions <- as( predict(rec, x, type = "ratingMatrix"), "data.frame") %>%
      mutate(userId = as.numeric(.$user),  #format generic dataframe names
             movieId = as.numeric(.$item), 
             residual_hat = rating) %>% 
      select(userId, movieId, residual_hat)
    
    # join the test data with the prediction data frame
    SVDrecs <- test %>% select(userId, movieId, residual) %>%
      filter( userId %in% users) %>%
      left_join(., predictions, by = c('userId', 'movieId')) %>%
      filter(!is.na(residual_hat))
    
    # And calculate the loss function, RMSE
    RMSE(SVDrecs$residual, SVDrecs$residual_hat)
  }
  
  # The available methods for real rating matrix types
  methods = c('SVD','SVDF','UBCF','IBCF','ALS')
  
  # See which method performs best on our small dataset
  recommenderRMSEs <- sapply(methods, function(method)
    CalcRecommenderRMSE(x, method))
  recommenderRMSEs
  
  # Clean up
  rm( methods, recommenderRMSEs )

#' 
#' We see that on the small subset of users, SVD had the best performance, and
#' again SVDF and ALS perform similarly well. We will proceed with tuning the
#' user and movie interaction SVD prediction in the same way that we tuned for
#' user genre interaction, except we'll tune it on the entire `train` set. We
#' will keep in mind that we might want to add in some of the features of the
#' Funk SVD algorithm as it was the next best performer and we selected `SVD`
#' based on its performance on a small subset of our `train` set.
#' 
#' 
## ----FullSVD, message=FALSE, error=FALSE-----------------------------------------------------
  # Before doing the following calculation, we need to do some cleanup
  # This calculation takes time and a lot of compute resources, so we need the 
  # minimum amount of extraneous data cluttering up the RAM
  rm( b_its, b_ugs, b_uts, movie_mus, user_mus)

  # Fill matrix with user vs movie residual rating and convert to recommenderlab
  # type realRatingMatrix
  x <- train %>% select(userId, movieId, residual) %>%
    spread(movieId, residual) %>%
    as.matrix() 
  rownames(x) <- x[,1] # create rownames
  x <- x[,-1]  # trim off userIds to leave just residuals
  x <- as( x, 'realRatingMatrix')  # make sparse
  
  # Map the users and movies in test set to the train set index values
  # the matrix factors are organized according to location in train set
  uniq_userIds <- train %>% group_by(userId) %>%
    summarize(userId = first(userId)) %>%
    pull(userId)

  uniq_movieIds <- train %>% group_by(movieId) %>%
    summarize(movieId = first(movieId)) %>%
    pull(movieId)

  # where do we find the test user and movie in the rec objects - same order as
  # train set.  U and V correspond to the matrix factors in UdV'
  U_indexs <- data.frame( userId = sort(unique(test$userId)),
                         U_index = which(uniq_userIds %in% test$userId))

  V_indexs <- data.frame( movieId = sort(unique(test$movieId)),
                         V_index = which(uniq_movieIds %in% test$movieId))

  # Add the indexes to test set for easy access and organization
  test <- test %>% left_join(U_indexs, by = 'userId') %>%
    left_join(V_indexs, by = 'movieId')
  
  # Clean up
  rm( uniq_userIds, uniq_movieIds, U_indexs, V_indexs )
  
  # Train SVD model 
  rec <- Recommender( x, 
                      method = 'SVD', 
                      list(k = 85, 
                           normalize = NULL, 
                           verbose = FALSE))
  
  #Do some cleanup, free up RAM before going on
  rm(x)

#'
#' We then calculate our prediction. Because of the size of the output matrix,
#' we do this element-wise and only calculate a prediction for a user/movie pair
#' found in the `test` set. The figure below shows the regularization curve used
#' to minimize $RMSE$ by the number of matrix factor components, or k's.
#' 
## ----TrainbuiSVD, cache=TRUE, eval = TRUE, warning=FALSE, error=FALSE, fig.height=3, fig.cap="Regularization Curve for SVD number of matrix factor components to minimize RMSE."----
  # Initialize a b_ui to build with a for loop, very un-R-like, but works
  b_ui_SVD <- rep(0,10)
  
  ks <- c(seq(10,80,10),85) # Values for number of components to use
  
  # loop through values of k to train for best RMSE
  rmses <- sapply( ks , function(k) {
    # Calculate b_ui element-wise
    for( i in 1:nrow(test)) b_ui_SVD[i] <- 
        sum( rec@model$svd$u[test$U_index[i], 1:k ]  * 
               rec@model$svd$d[1:k] *
               rec@model$svd$v[test$V_index[i], 1:k ])

    RMSE(test$residual,b_ui_SVD) # RMSE calculated for each k
  })
  
  # Plot the k that minimizes RMSE
  qplot(ks,rmses, main="User Movie Interaction Components to Minimize RMSE")
  
  # select best value of k and save it
  K_User_Movie_SVD <- ks[which.min(rmses)]
  
  # display best RMSE and the value of k to obtain it
  K_User_Movie_SVD
  min(rmses)

#' 
#' # Results
#' 
## ----AnalysisResults, echo=FALSE-------------------------------------------------------------
  # Clean up previous chunk
  rm(i, b_ui_SVD)

  # Add it to the results
  results <- results %>% 
    add_row(method = "User/movie SVD", 
          RMSE = min(rmses))
  
  # Make a nice table
  knitr::kable(results)

#' 
#' ## Train the Final Model
#' ### What Is in the Final Model
#' The blend of effects we will use to train the final model includes the
#' following effects:
#' 
## ----EffectsInBlend, echo = FALSE, cache=TRUE------------------------------------------------
knitr::kable(data.frame( Effect = c("$\\mu$", "$b_{i}$", "$b_{u}$",
                                    "$b_{i,t}$", "$b_{u,t}$", "$b_{u,g}$",
                                    "$b_{i,u}$"),
                         Description = c("The overall average rating",
                                    "Average rating for each movie",
                                    "Average rating for each user",
                                    "Average rating by movie and age in years",
                                    "Average rating by user and hour of day",
                                    "User-genre interaction by matrix factorization",
                                    "User-movie interaction by matrix factorization") ),
             align = c('l','l'),
             format = 'latex',
             linesep = "",
             caption = 'Effects Included in the Final Model',
             escape=FALSE)

#'
#' So the final model is a linear combination of the effects as in the following
#' equation:
#'
#' $$ \hat{y} = \hat{\mu} + \hat{b}_{i} + \hat{b}_{u} + \hat{b}_{i,t} +
#' \hat{b}_{u,t} + \hat{b}_{u,g} + \hat{b}_{u,i} $$
#'
#' where the hats indicate our estimate of the actual rating and effect
#' components, and as we have done in previous sections, user is indicated with
#' a $u$ subscript and movie is indicated with an $i$ subscript. The $t$
#' subscript is used to indicate temporal effects, but the two time-dependent
#' effects are not on the same time-scale. For $\hat{b}_{i,t}$ the scale of the
#' time variable is in years since release, while for $\hat{b}_{u,t}$ the time
#' variable is in hour of the day based on a twenty four hour clock.
#'
#' The final equations for each effect are given below. $$ \hat{\mu} =
#' \frac{1}{N_{y}}\sum_{}y$$ $$\hat{b}_{i} = \frac{1}{n_{i} +
#' \lambda_i}\sum_{u=1}^{n_i}( y_{u,i} - \hat{\mu})$$ $$\hat{b}_{u} =
#' \frac{1}{n_{u} + \lambda_{u}}\sum_{i=1}^{n_u}( y_{u,i} - \hat{\mu} -
#' \hat{b}_{i})$$ $$\hat{b}_{i,t} = \frac{1}{n_{i,t} +
#' \lambda_{i,t}}\sum_{u=1}^{n_{i,t}}( y_{u,i,t} - \hat{\mu} - \hat{b}_{i} -
#' \hat{b}_{u})$$
#'
#' $$\hat{b}_{u,t} = \frac{1}{n_{u,t} + \lambda_{u,t}}\sum_{i=1}^{n_{u,t}}(
#' y_{u,i,t} - \hat{\mu} - \hat{b}_{i} - \hat{b}_{u} - \hat{b}_{i,t})$$ We then
#' define a residual, $\hat{r}_{u,i}$ as $$ \hat{r}_{u,i,t} = y_{u,i,t} -
#' \hat{\mu} - \hat{b}_{i} - \hat{b}_{u} - \hat{b}_{i,t} - \hat{b}_{u,t}$$ And
#' we assign each `genre` in the list a number from one to the number of genre
#' combinations in the training set. We know that we will include all possible
#' `genre` combinations for the `test` set because we chose our test set such
#' that all `userIds` and `movieIds` from the test set were represented in the
#' training set, and the `genre` combination has a 1:1 correlation with
#' `movieId`. In other words, all movies with the same `movieId` have the same
#' `genres` combination. This residual was calculated from the residual in the
#' same manner as our other baseline predictors, then regularized, and the
#' result was decomposed into its matrix components with the Funk variant of
#' Singular Value Decomposition (SVD). $$\hat{b}_{u,g} \approx p_uq_i$$ where
#' the values of the matrix factors are further approximated as $$\hat{b}_{u,g}
#' = \sum_{k=1}^{K} p_{u,k}q_{i,k}$$ We choose the number of singular vector
#' pairs to use, or the value of $K$, through training, minimizing $RMSE$.
#'
#' We then re-calculate the residual to be $$ \hat{r}_{u,i,t} = y_{u,i,t} -
#' \hat{\mu} - \hat{b}_{i} - \hat{b}_{u} - \hat{b}_{i,t} - \hat{b}_{u,t} -
#' \hat{b}_{u,g}$$ and approximate the interaction between users and movies, or
#' $\hat{b}_{u,i}$ with a second set of decomposed vector pairs. This time our
#' training indicated a better result with just the SVD method rather than the
#' more complex Funk variant, so we get $$\hat{b}_{u,i} = \sum_{j=1}^{J}
#' p_{u,j}q_{i,j}$$
#'
#' The final prediction was made by applying the clipping function, $\kappa()$.
#' $$ \hat{y}_{u,i,t} = \kappa \Big\rvert_{0.5}^{5.0}( \hat{\mu} + \hat{b}_{i} +
#' \hat{b}_{u} + \hat{b}_{i,t} + \hat{b}_{u,t} + \hat{b}_{u,g} + \hat{b}_{u,i} )
#' $$
#' 
#' ### Training the Model
#' We deconstructed the `edx` set to do our exploration and initial training, so
#' we now need to reconstitute it.
## ----edxReconstitution, warning=FALSE, message=FALSE, cache=TRUE-----------------------------
  # Re-construct edx after our data exploration and model building
  test <- test %>% select(userId, movieId, rating, timestamp, title, year, 
                          genres, age, hour)
  train <- train %>% select(userId, movieId, rating, timestamp, title, year, 
                          genres, age, hour)
  edx <- full_join(test, train,
                   by = c('userId', 'movieId', 'rating', 'timestamp', 'title',
                          'year', 'genres', 'age', 'hour'))
  
  rm(test,train)

#' 
#' Then we can begin to calculate the various elements of the blend, starting
#' with the estimated mean, $\hat{\mu}$. For all of the training parameters that
#' are not based on matrix factorization, we will use cross-validation with the
#' `edx` train:test ratio at 9:1.
## ----TrainFullModel, echo=FALSE, cache=TRUE, message=FALSE, warning=FALSE--------------------
  
  # Function to train the model's non-matrix-factorization parameters
  TrainFullModel <- function(){
    
    # create new partition/cross validation set for each call
    test_index_temp <- createDataPartition(y = edx$rating, times = 1, 
                                           p = 0.1, list = FALSE)
    train_temp <- edx[-test_index_temp, ]
    temp <- edx[test_index_temp, ]
    
    # Make sure userId and movieId in test set are also in train set
    test_temp <- temp %>% 
        semi_join(train_temp, by = 'movieId') %>%
        semi_join(train_temp, by = 'userId')
    
    # Add rows removed from validation set back into edx set
    removed <- anti_join(temp, test_temp)
    train_temp <- rbind(train_temp, removed)
    
    # ----   Calculate mean ---------------
    mu <- mean(train_temp$rating)
    
    # ----   Calculate b_i and L_i --------
    lambdas <- seq(2.0,4,0.25)
    regularized_rmses <- sapply(lambdas, function(L){
      movie_mus <- train_temp %>%
        group_by(movieId) %>%
        summarize(b_i = sum(rating - mu)/(n() + L))
      
      predicted_ratings <- test_temp %>%
        left_join(movie_mus, by = 'movieId') %>%
        mutate(prediction = mu + b_i) %>%
        pull(prediction)
      
      return( RMSE(test_temp$rating, predicted_ratings))
    })
    
    error <- min(regularized_rmses)  # ~0.9441152
    L_i <- lambdas[which.min(regularized_rmses)]  #~1.75
    
    # final trained b_i and L_i
    movie_mus <- train_temp %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n() + L_i))
    
    # ----   Calculate b_u and L_u --------------
    lambdas <- seq(3.5,5,0.25)
    regularized_rmses <- sapply(lambdas, function(L){
      user_mus <- train_temp %>%
        left_join(movie_mus, by = 'movieId') %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - mu - b_i)/(n() + L))
      
      predicted_ratings <- test_temp %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        mutate(prediction = mu + b_i + b_u) %>%
        pull(prediction)
      
      return( RMSE(test_temp$rating, predicted_ratings))
    })
    
    error <- min(regularized_rmses)  
    L_u <- lambdas[which.min(regularized_rmses)]  #~4.25
    
    # final trained b_u and L_u
    user_mus <- train_temp %>%
      left_join(movie_mus, by = 'movieId') %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - mu - b_i)/(n() + L_u))
    
    # ----- Train b_it and L_it ------------------
    lambdas <- seq(20,50,5)
    regularized_rmses <- sapply(lambdas, function(L){
      b_its <- train_temp %>%
        mutate( age = year(timestamp) - year) %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        group_by(movieId,age) %>%  # movieId AND age, not just age
        summarize(b_it = sum(rating - mu - b_i - b_u)/(n() + L))

      predicted_ratings <- test_temp %>%
        mutate( age = year(timestamp) - year) %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        left_join(b_its, by = c('movieId','age') ) %>%
        mutate(prediction = mu + b_i + b_u + nafill(b_it,fill=0)) %>%
        pull(prediction)
      return( RMSE(test_temp$rating, predicted_ratings))
    })

    error <- min(regularized_rmses)  #~ 0.8601591
    L_it <- lambdas[which.min(regularized_rmses)]

    # Final trained b_its and L_it
    b_its <- train_temp %>%
      mutate( age = year(timestamp) - year) %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      group_by(movieId,age) %>%  # movieId AND age, not just age
      summarize(b_it = sum(rating - mu - b_i - b_u)/(n() + L_it))

    # # ---- Train b_ut and L_ut -----------------
    lambdas <- seq(0,50,5)
    regularized_rmses <- sapply(lambdas, function(L){
      b_uts <- train_temp %>% mutate( hour =  hour(timestamp),
                                      age = year(timestamp) - year) %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        left_join(b_its, by = c('movieId','age')) %>%
        group_by( userId, hour ) %>%
        summarize( b_ut = sum(rating - mu - b_i - b_u - b_it)/(n() + L) ) %>%
        select( userId, hour, b_ut)

      predicted_ratings <- test_temp %>%
        mutate( age = year(timestamp) - year, hour = hour(timestamp)) %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        left_join(b_its, by = c('movieId','age')) %>%
        left_join(b_uts, by = c('userId', 'hour')) %>%
        mutate(prediction = mu + b_i + b_u +
                 nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
                 nafill(b_ut, fill=0)) %>%
        pull(prediction)
      return( RMSE(test_temp$rating, predicted_ratings))
    })

    error <- min(regularized_rmses)  
    L_ut <- lambdas[which.min(regularized_rmses)]

    #Final trained b_ugs and L_ug
    b_uts <- train_temp %>% mutate( hour =  hour(timestamp),
                                      age = year(timestamp) - year) %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        left_join(b_its, by = c('movieId','age')) %>%
        group_by( userId, hour ) %>%
        summarize( b_ut = sum(rating - mu - b_i - b_u - b_it)/(n() + L_ut)) %>%
        select( userId, hour, b_ut)

    # -------- Train b_ug and L_ug -----------------
    lambdas <- seq(7,11,0.5)
    regularized_rmses <- sapply(lambdas, function(L){
      b_ugs <- train_temp %>% 
        mutate( hour =  hour(timestamp),
                age = year(timestamp) - year,
                genres = as_factor((genres))) %>%
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        left_join(b_its, by = c('movieId','age')) %>%
        left_join(b_uts, by = c('userId','hour')) %>%
        group_by(userId, genres) %>%
        summarize(b_ug = sum(rating - mu - b_i - b_u - b_it - b_ut)/(n() + L))

      predicted_ratings <- test_temp %>%
        mutate( age = year(timestamp) - year, 
              hour = hour(timestamp),
              genres = as_factor(genres)) %>%        
        left_join(movie_mus, by = 'movieId') %>%
        left_join(user_mus, by = 'userId') %>%
        left_join(b_its, by = c('movieId','age')) %>%
        left_join(b_uts, by = c('userId', 'hour')) %>%
        left_join(b_ugs, by = c('userId', 'genres')) %>%
        mutate(prediction = mu + b_i + b_u + 
                 nafill(b_it, fill = 0) +   #fill NA with 0, ie no effect
                 nafill(b_ut, fill = 0) +
                 nafill(b_ug, fill = 0)) %>%
        pull(prediction)
      return(RMSE(test_temp$rating, predicted_ratings))
    })

    error <- min(regularized_rmses)
    L_ug <- lambdas[which.min(regularized_rmses)]  #~8.5
    
    b_ugs <- train_temp %>% 
      mutate( hour =  hour(timestamp),
              age = year(timestamp) - year,
              genres = as_factor((genres))) %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      left_join(b_its, by = c('movieId','age')) %>%
      left_join(b_uts, by = c('userId','hour')) %>%
      group_by(userId, genres) %>%
      summarize(b_ug = sum(rating - mu - b_i - b_u - b_it - b_ut)/(n() + L_ug))
    
    
        # ------- Clean up and return parameters --------
    rm(temp, test_temp, train_temp, removed)
    return(data.frame("mu" = mu, "error" = error, "L_i" = L_i,
                      "L_u" = L_u, "L_it" = L_it, "L_ut" = L_ut, 
                      "L_ug" = L_ug))
  }

#' 
## ----crossValidateFinal, cache=TRUE, message=FALSE, warning=FALSE----------------------------
  # Re-set the seed before we begin the final training
  set.seed(1234, sample.kind="Rounding")

  # Cross-validate regularized elements B times
  B <- 5
  cv_results <- replicate(B, TrainFullModel() )
  
  # transform the cross-validation results into usable data frame
  cv_results <- as.data.frame(t(matrix(unlist(cv_results), ncol=B)))

#'
#' We use the parameters trained on the full `edx` set to create the individual
#' components of the final prediction and create a `residual` for the `edx` set.
#' 
## ----buildLinearPredictionPieces, echo = FALSE, cache=TRUE-----------------------------------
  # Save the previously trained parameters for use in the final model  
  mu <- mean(edx$rating)
  L_i <- mean(cv_results$V3)
  L_u <- mean(cv_results$V4)
  L_it <- mean(cv_results$V5)
  L_ut <- mean(cv_results$V6)
  L_ug <- mean(cv_results$V7)

  # Calculate final movie effect, b_i
  movie_mus <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + L_i))

  # Calculate final user effect, b_u
  user_mus <- edx %>%
    left_join(movie_mus, by = 'movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + L_u))
  
  # Calculate final movie temporal effect, b_it
  b_its <- edx %>%
    mutate( age = year(timestamp) - year) %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    group_by(movieId,age) %>%  # movieId AND age, not just age
    summarize(b_it = sum(rating - mu - b_i - b_u)/(n() + L_it))
    
  # Calculate final user temporal effect, b_ut
  b_uts <- edx %>% 
    mutate( hour =  hour(timestamp), age = year(timestamp) - year) %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      left_join(b_its, by = c('movieId','age')) %>%
      group_by( userId, hour ) %>%
      summarize( b_ut = sum(rating - mu - b_i - b_u - b_it)/(n() + L_ut)) %>%
      select( userId, hour, b_ut)
    
  # Add the completed elements to edx and recalculate the residual
  # this residual becomes the input to the matrix factorization steps
  edx <- edx %>%
      mutate( hour =  hour(timestamp), age = year(timestamp) - year) %>%
      left_join(movie_mus, by = 'movieId') %>%
      left_join(user_mus, by = 'userId') %>%
      left_join(b_its, by = c('movieId','age')) %>%
      left_join(b_uts, by = c('userId','hour')) %>%
      mutate(residual = rating - mu - b_i - b_u - b_it - b_ut )
    
  # The first step in making the validation prediction - add the pieces we
  # have completed - we can calculate the prediction when we have completed
  # the two matrix factorization steps and added those elements to validation
  validation <- validation %>% 
    extract(data = . , 
            col = title, 
            into = c("title", "year"), 
            regex = "(.+)[(](\\d+)[)]" )%>% 
    mutate(timestamp = lubridate::as_datetime(timestamp),
           year = as.numeric(year))

  validation <- validation %>%
    mutate( hour =  hour(timestamp), 
            age = year(timestamp) - year) %>%
    left_join(movie_mus, by = 'movieId') %>%
    left_join(user_mus, by = 'userId') %>%
    left_join(b_its, by = c('movieId','age')) %>%
    left_join(b_uts, by = c('userId','hour')) 
    
    

#' 
#' 
## ----ListOfParams, echo=FALSE, cache=TRUE----------------------------------------------------
  knitr::kable(data.frame( Parameter = c("L_i", "L_u", "L_it", 
                                         "L_ut", "L_ug"),
                            Value = c(L_i, L_u, L_it, L_ut, L_ug) ),
                align = c('l','l'),
                format = 'latex',
                linesep = "",
                caption = "Final Model Parameters for Regularized Elements")

#'
#' We have now trained all parameters of the model up to the matrix
#' factorization models. The matrix factorization models are too computationally
#' intensive to perform multiple times, though we would likely improve our
#' results with cross-validation. However, with limitations in our computing
#' resources, we will use the previously trained parameters for the final, full
#' `edx` model.
#' 
## ----edxSVDF, cache=TRUE, echo=TRUE, message=FALSE, error=FALSE------------------------------
  # Calculate regularized average rating per user, per genre combination 
  # as residual of previously calculated effects
  b_ugs <- edx %>% 
      mutate( genres = as_factor((genres))) %>%
      group_by(userId, genres) %>%
      summarize(b_ug = sum(residual)/(n() + L_ug))

  # Construct realRatingMatrix of user vs factorized genre combinations
  # filled with residual average rating for that genre combo
  x <- b_ugs %>%
    spread(genres, b_ug) %>% 
    as.matrix()
  rownames(x) <- x[ ,1]
  x <- x[ ,-1]
  x_pred <- x[ rownames(x) %in% validation$userId , ]
  x <- as(x, 'realRatingMatrix')
  x_pred <- as(x_pred, 'realRatingMatrix')

  # Calculate the SVDF model for genre/user interaction
  rec <- Recommender( x, method = 'SVDF', 
                      list(k = kGenreSVDF, 
                         gamma = GammaGenreSVDF,
                         verbose = FALSE,
                         normalize = NULL))
  

  # Make a prediction to fill in missing data
  b_ugs <- as( predict(rec, x_pred, type = "ratingMatrix"), "data.frame") %>%
    mutate(userId = as.numeric(.$user),  #format generic dataframe names
           genres = item , 
           b_ug = rating) %>% 
    select(userId, genres, b_ug)
  
  # Add the predicted b_ug to edx and validation sets and recalculate residual
  # in the edx set - residual is input to movie/user SVD below
  edx <- edx %>%
    mutate(genres = as_factor(genres)) %>%
    left_join(b_ugs, by = c('userId','genres')) %>%
    mutate(residual = residual - b_ug)
    
  # Add b_ug to validation for later incorporation into the final prediction
  validation <- validation %>%
    mutate(genres = as_factor(genres)) %>%
    left_join(b_ugs, by = c('userId','genres'))
  
  # Clean up
  rm(b_its, b_ugs, b_uts, cv_results, movie_mus, rec, 
     user_mus, x, x_pred, predicted_ratings, i)

#' 
#' We can now calculate the SVD matrix factorization components.  
#' 
## ----SVDFull, echo=TRUE, message=FALSE, warning=FALSE,error=FALSE----------------------------
  # Create realRatingMatrix of userId vs movieId, filled with rating residuals
  x <- edx %>% select(userId, movieId, residual) %>%
    spread(movieId, residual) %>%
    as.matrix() 
  rownames(x) <- x[,1] # create rownames
  x <- x[,-1]  # trim off userIds to leave just residuals
  x <- as( x, 'realRatingMatrix') # convert to recommenderlab type
  
  # Train SVD model. We can optimize with just one calculation if K is large
  rec <- Recommender( x, 
                      method = 'SVD', 
                      list(k = 85, 
                           normalize = NULL, 
                           verbose = FALSE))
  
  #Do some cleanup, free up RAM before going on
  rm(x)

#' 
## ----Train_b_ui_SVDFinal, eval = TRUE, cache=TRUE--------------------------------------------
  # Create lists of unique users and movies in edx
  uniq_userIds <- edx %>% group_by(userId) %>%
    summarize(userId = first(userId)) %>%
    pull(userId)

  uniq_movieIds <- edx %>% group_by(movieId) %>%
    summarize(movieId = first(movieId)) %>%
    pull(movieId)

  # Get index locations for each movie/user in validation set - map to location
  # in edx set because output of SVD is in that order
  U_indexs <- data.frame( userId = sort(unique(validation$userId)),
                           U_index = which(uniq_userIds %in% validation$userId))

  V_indexs <- data.frame( movieId = sort(unique(validation$movieId)),
                           V_index = which(uniq_movieIds %in% validation$movieId))

  # Add the location for each movie/user to the validatio set for easier access
  validation <- validation %>% left_join(U_indexs, by = 'userId') %>%
      left_join(V_indexs, by = 'movieId')
  
  # Clean up
  rm( uniq_userIds, uniq_movieIds, U_indexs, V_indexs )

  # Calculate b_ui, element-wise, only for movie/user pairs in validation
  # we only use 1:K components of the factorization, per trained K value to 
  # minimize RMSE
  b_ui_SVD <- rep(0,10)
  for( i in 1:nrow(validation)) b_ui_SVD[i] <- 
      sum( rec@model$svd$u[validation$U_index[i], 1:K_User_Movie_SVD ]  *
            rec@model$svd$d[1:K_User_Movie_SVD] *
            rec@model$svd$v[validation$V_index[i], 1:K_User_Movie_SVD ])

  # Add completed b_ui to validation for later addition to final prediction
  validation <- validation %>%
    mutate(b_ui = b_ui_SVD)

#' # Final RMSE
#' We can now calculate our final loss function value,
## ----FinalRMSE, cache=TRUE-------------------------------------------------------------------
  # Add the final elements so we can calculate prediction
  validation <- validation %>%
    mutate(prediction = mu + b_i + b_u + 
             nafill(b_it, fill = 0) + 
             nafill(b_ut, fill = 0) + 
             nafill(b_ug, fill = 0) + 
             b_ui) %>%
    mutate(prediction = SimpleClip(prediction, UB = 5.0, LB = 0.5))

  RMSE(validation$rating, validation$prediction) #0.8228107

#'
#'
#'
#' # Conclusion
#'
#' The recommender system we constructed had a final $RMSE$ of $0.8228107$. The
#' final model was trained with $5x$ cross-validation on the entire `edx` set,
#' with regularized baseline effects for average rating for each user, average
#' rating for each movie, average rating for each movie in each year after its
#' release date, average rating for each user for each hour of the day, and
#' average rating for each user for each genre combination, each regularized to
#' penalize low sample size. User/genre interaction was further predicted with
#' the Simon Funk variant of Singular Value Decomposition (FSVD or SVDF) with
#' the user/genre averages as the input values and `recommenderlab`'s default
#' training parameters except for our trained values of $\gamma = 0.025, K =
#' 12$. The final piece of the recommender was the user/movie interaction, which
#' we estimated with straight Singular Value Decomposition and a trained value
#' of $K = 80$. Both matrix factorization models were trained with $1x$
#' cross-validation on `edx`. The final prediction was coerced to the known data
#' range for ratings of $\{0.5,5.0\}$ by just assigning values that were out of
#' range to the nearest in-bounds value. We refer to this clipping function as
#' $\kappa()$ in the text.
#'
#' While the final $RMSE$ of $0.8228107$ was a very good result, there is still
#' plenty of room for improvement. There were multiple baseline effects that we
#' could have pursued further, such as user rating drift over time and sudden
#' changes in rating frequency. Additionally, we could do more training on the
#' two matrix factorization elements, and we chose not to do the extra training
#' due to the computing resources required to calculate and evaluate multiple
#' large matrices. The initial work on this analysis was done on a system with
#' 8GB of RAM and a Funk SVD model that was trained early in the process took
#' more than eight hours to run on that machine and crashed the `R` session if
#' cleanup was not diligently performed before the calculation was attempted.
#' The rest of the analysis was done on a platform with 62 Gigabytes of RAM. We
#' note that just the `rec` object that is the output of the user/movie matrix
#' factorization operation with `recommenderlab` is more than 55 Megabytes and
#' multiple variables produced in this model are in excess of 1GB each. In
#' short, with these large data objects, swap space and the time required to
#' swap data becomes limiting. If we were going to do this analysis on a
#' platform similar to a small laptop, we would need to explore other ways to
#' handle large data, such as keeping our variables on disk rather than in RAM
#' and working with just small chunks at a time. `R` does have multiple tools
#' for handling these types of problems, but they weren't really covered in the
#' course material and we didn't explore them in this analysis.
#'
#' We also could have explored more of the neighborhood and clustering models
#' presented in the course material. We noted that the `R` implementation of
#' k-nearest neighbors will throw an error when we attempt to do an analysis
#' with more than fifty five levels in a factorized response variable. A search
#' through online forums suggests a solution to this problem is to edit the
#' source code and recompile from source. We concluded that this was beyond the
#' scope of the current analysis, even though neighborhood techniques seem to be
#' an intuitively good fit for this type of recommender system. It seems that we
#' should be able to cluster users and movies into groups with similar ratings
#' and if users Billy and Bob have similar ratings for many movies, and Billy
#' has rated a movie highly that Bob has not rated, then Bob is likely to also
#' rate that movie highly. In essence, we are performing a similar task with
#' matrix factorization, but it initially seemed that Random Forest or K-Nearest
#' Neighbor techniques could also work well. We note that all of the Netflix
#' Prize solutions we have referenced dealt with matrix factorization techniques
#' and opted not to use decision tree techniques.
#'
#' One interesting result was the lack of performance-boosting from our clipping
#' techniques. The Funk SVD method uses a sigmoid style of clipping to shrink
#' the predictions back from the outer bounds of the prediction space, and this
#' seems to make sense from a logical perspective. We noted that our predicted
#' ratings contained out-of-bounds values after just the calculation of the user
#' and movie effects, and by feeding the out-of-bounds residuals into the
#' calculation of the next effect, we should have been amplifying our error.
#' However, we noted that both of the clipping methods we tried, even with
#' training, did not improve our performance very much, if at all, and in some
#' cases the clipping procedure contributed very negatively to performance. This
#' was the justification for only applying clipping as the final step, when it
#' would be mathematically guaranteed to improve performance because we do know
#' the range of ratings in the `validation` set, though we know very little else
#' about the ratings in that set. We did not test or train intermediary clipping
#' on all elements of the model, however, so more work could be done to
#' ascertain if intermediary clipping might have been beneficial in some cases.
#'
#' Overall, this was a good exercise in the construction of a recommender system
#' with matrix factorization. We have demonstrated that matrix factorization is
#' a very powerful and elegant tool and is especially useful when we have
#' missing data in a relational matrix, such as the movie-user matrix we
#' constructed in the analysis. We further demonstrated that matrix
#' factorization is very powerful when we have large datasets because it allows
#' us to reduce dimensionality while simultaneously making a large number of
#' predictions for the missing data. In all, matrix factorization is a wonderful
#' tool to add to our machine learning toolbox.
