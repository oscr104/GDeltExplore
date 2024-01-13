This project will be tools to query, visualise and analyse data on right wing extremist groups from GDelts (https://www.gdeltproject.org/) global events databases / knowledge graphs

Eventually these tools will fit together as an interactive app to show different groups by area, affiliation, time period etc. 

Currently working on exploratory notebooks to work up analysis methods/strategies

-------------------------------------------------------------------------------
How to run:

You need a Google Cloud account with Big Query set-up (https://cloud.google.com/bigquery/docs/introduction) and enough credits/budget to run the queries (you get $300 for free on sign up, for 90 days)

Set up a new conda environment, e.g. "conda create -n kye_env", activate it "conda activate kye_env" and install requirements "conda install --file requirements.txt

Once you've set that up, make sure you have a project set up in Google Cloud for you to query data to, run "gcloud auth login" to set up your connection to the cloud, and run the explore.ipynb notebook