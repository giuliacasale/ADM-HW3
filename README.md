# Homework 3 of the course in algorithmic methods in data mining  master degree course in data science at Sapienza University in Rome  A.Y. 2020/2021 #


![](https://github.com/giuliacasale/ADM-HW3/blob/main/06critics-list1-videoSixteenByNineJumbo1600.jpg)

**the goal of this homework is to perform some ordinary tasks in text mining, such as: web scraping, html parsing, search engines building and similarity evaluations. For this Homework it has been provided no dataset, but the last one has to be built from scratch by obtaining information from the following site: https://www.goodreads.com, where it is possible to find a ranking of the most popular books published until now. For each book there is a web page like the following: https://www.goodreads.com/book/show/2767052-the-hunger-games, which shows different information about the book, as number of pages, ratings and plot.**

**the assigment is divided into five different parts.** 

1. the first part deals with web scraping and html parsing. This part is related to dataset building which is performed, firstly by downloading the html pages of the first 30.000 books in the ranking and secondly by parsing the obtained pages in order to get the following information: Title, Series, Author(s), Ratings, number of given ratings, number of reviews, plot, number of pages, published date, characters, setting and url. These will be the columns of the dataset, while each row represents a specific book.

2. in the second part,  there are implemented two search engines. The first engine, given a query inserted by the user, returns, among all the books in the dataset, only the documents which include in the plot all the words in the query. The second search engine, instead, returns the top_k documents, ordered by similarity ( measured with cosine similarity) which incluedes, as first search engine does, all the query words. 

3. In the third part, it has been formulated a new score measure which takes into account some of the variable included in the dataset.  The top-k documents are 
ordered by adopting this new measure.

4. In the fourth part (bonus question) is provided visualization of some book series, with the aim to inform about writer's production during the years.

5. The fifth part (theory question) is not strictly related to the task, but it consists in an algorithmic question about finding the maximum lenght of subsequence of characters that are in alphabetical order by exploiting both recursive algorithms and dynamic programming, showing the complexity differences between these two approaches. 

as well as this markdown file, the content of this repository includes HM3.ipynb file, that is a jupyter notebook in which our group has answered to the assigment questions, providing both code in python and explanations regarding the accomplished operations. a function.py file, where it is possible to find the single functions, separated for parts, used to perform a specific task objectives. a DataCollection.py file, where there is the code used to perform part one. Finally, there is Homework_3_algorithm_question.ipynb where the theory question about algorithms is answered.
