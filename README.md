# Homework 3 of the course in algorithmic methods in data mining  master degree course in data science at Sapienza University in Rome  A.Y. 2020/2021 #


![](https://github.com/giuliacasale/ADM-HW3/blob/main/06critics-list1-videoSixteenByNineJumbo1600.jpg)

### The goal of this homework is to perform some ordinary tasks in text mining, such as: web scraping, html parsing, search engines building and similarity evaluations.

**For this Homework it has been provided no dataset, but it has to be built from scratch by obtaining information from the following site: https://www.goodreads.com, where it is possible to find a ranking of the most popular books published until now. For each book there is a web page like the following: https://www.goodreads.com/book/show/2767052-the-hunger-games, which shows different information about the book, as number of pages, ratings and plot.**

## The assigment is divided into five different parts.

1. The first part deals with web scraping and html parsing. This part is related to dataset building. firstly, the urls of the first 300 pages of GoodReads ranking (in each html page of the ranking there are 100 links which refers to the html book pages), are taken and stored in a txt file. Secondly, the html pages of the urls previously stored are downloaded. Afterwards, it is done parsing of the downloaded html pages, in order to get the following information: Title, Series, Author(s), Ratings, number of given ratings, number of reviews, plot, number of pages, published date, characters, setting and url. These will be the columns of the dataset, while each row represents a specific book. This dataset will be the basis for the following parts.

2. In the second part,  there are implemented two search engines. The first engine, given a query inserted by the user, returns, among all the books in the dataset, only the documents which include in the plot all the words in the query. The second search engine, instead, returns the top_k documents, ordered by similarity ( measured with cosine similarity) which incluedes, as first search engine does, all the query words. 

3. In the third part, it has been formulated a new score measure which takes into account some of the variables included in the dataset. This score has been called user-score and it is the result of a weighted average (with weights determined by the user) of the cosine similarity (as in part 2), the ranking score (new score measure), the lenght_score (new score measure), the published_score (new score measure). The top-k documents are sorted by the user_score. The user score and the other score measures quoted above are displayed.

4. In the fourth part (bonus question) is provided visualization of some book series, with the aim to inform about writer's production during the years.

5. The fifth part (theory question) is not strictly related to the task, but it consists in an algorithmic question about finding the maximum lenght of subsequence of characters that are in alphabetical order by exploiting both recursive algorithms and dynamic programming, showing the complexity differences between these two approaches. 

## As well as this markdown file, the content of this repository includes:
* HM3.ipynb file, that is a jupyter notebook in which our group has answered to the assigment questions, providing both code in python and explanations regarding the accomplished operations. 
* function.py file, where it is possible to find the single functions, separated for parts, used to perform a specific task objectives. 
* DataCollection.py file, where there is the code used to perform the download of the html files and their parsing. 
* finally, Homework_3_algorithm_question.ipynb provides the answer to the fifth part, there you can find the implementation of the algorithms, the demostrations of their complexity and some tests showing the differences between the two approaches. 
