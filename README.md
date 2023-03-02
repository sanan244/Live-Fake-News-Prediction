# Live-Fake-News-Prediction
This project aims to predict if the news on any week by CNN is real or fake. 
The source information is pulled from the weekly transcipts published on cnn's website.( https://transcripts.cnn.com/ )

## Code Review
The initial model that inspired this project can be found at https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/.

# Whats different?
What makes this repository unique is that we scrape our own real time data to predict on our model.
In order to do this I had to accurately access links within the source website, then access links within that source, and compile all of the weeks 
transcipts into a single dataset. Then I needed to run the data through the predictor and fine tune the model until results were sufficient.
