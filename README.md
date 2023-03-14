# Completed: Live-Fake-News-Prediction
This project aims to predict if the news on any current week by CNN is real or fake. 
The source information is pulled from the weekly transcipts published on cnn's website.( https://transcripts.cnn.com/ )

## Code Review
The initial model that inspired this project can be found at https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/ 
The data set "new.csv" to train the initial model can be downloaded through the tutorial link.

# Whats different?
What makes this repository unique is that we scrape our own real time data to predict on our model.
In order to do this I had to :
1. accurately access the correct links within the source website, then collect data within those sub-links
2. Preprocess all this data into a finalized dataset
3. Test our already trained model on our real world data
4. Finally, Verify and validate our results!

