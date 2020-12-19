# 595-final-project

FE595 final project.py is a python file for processing the tweets you select with the tag and time period you want. It will return a basic token analysis of the
tweets you select(In our project, we give an instance of #dowjones, from 2015/12/01-2020/12/01). Build a LDA model by the tweets and visualize. Finally, we do a
sentiment analysis. Computing the sentiment score and build a model of compound sentiment score & daily DJIA close price. Then, have a prediction by this model 
and compared with the actual price.

## Usage

```python
 
# Set Input
# the query you want to search on twitter
twitterquery = 'from:#dowjones since:2015-12-01 until:2020-06-01'
# #dowjones is the tag you interested in, since and until is the time period you want

ric = 'DJIA'  # the RIC of the equity/ETF/index you want
# the date you select
start_date = '2015-12-01'
end_date = '2020-06-01'

