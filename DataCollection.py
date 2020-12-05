from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import requests
import re
import os
from langdetect import detect

###### Download urls ######
def url_request():
    #creating the file
    f=open("./data/books_url.txt","w") 
    f.close()
    # extracting the necessary urls from the html page 
    # cycle for the first 300 pages
    for idx in range(0,300):
    
        page=requests.get("https://www.goodreads.com/list/show/1.Best_Books_Ever?page="+str(idx))
    
        main=BeautifulSoup(page.text, features="lxml")

        links = main.find_all('a') 
        link_list=[]
        for link in links:
    
            full_link = link.get("href")
            link_list.append(full_link)
    
        url_list=[]    

        for element in link_list[100:]:
            element=str(element)
            c=re.search("/book/show/",element)
            if c is not None:
                url_list.append(element)
        url_list1=[] 


        for i in range(0,len(url_list),2):
            
            url_list1.append(url_list[i])

 

        site_url="https://www.goodreads.com"
        books_url=[]
        for url_book in url_list1[0:-2]:
            book=site_url+url_book
            book=book.strip()
            book=book+"\n"
            
            books_url.append(book)
            #append each book url to the file created before
            f=open("./data/books_url.txt","a") 
            f.write(book) 
            f.close()
    
    return


###### Download pages ######

def download_pages(urls_file="./data/books_url.txt", starting_index=0):
    # open the file with urls
    file1=open(urls_file,"r")
    c=file1.readlines()
    d=[]
    # appending to an empty list
    for string in c:
        d.append(string.rstrip("\n"))

    i=starting_index  # this value has to be initialized based on the starting point of d
    driver = webdriver.Firefox(executable_path=GeckoDriverManager() .install())  # if you have not firefox it is not ok

    for url in d:  # d is a list with lenght equal to 30.0000 if you select d[10000:20000] you will do only that subset
        print(i)
        # using selenium to bypass the block of the site
        driver.get(url)
        code=driver.page_source
        
        f=open("./data/page_1/book_"+str(i)+".html","w") 
        f.write(code) 
        f.close()
        i=i+1
    
    
###### Parse downloaded pages ######

def from_list_to_str(lis):
    return ' ; '.join(lis)


def clean_str(string):
    string = string.strip().replace('\n', ' ').replace('\t', ' ')  # Remove blank spaces, end line and tab
    if not string:  # If the string is empty return
        return ''
    if string[-1] == ',':  # If the last string's char is a comme remove it
        string = string[:-1]
    return ' '.join(string.split())  # Remove blank spaces


def get_publication_date(publication_string):
    word_list = publication_string.split()
    index = 1
    for i in range(len(word_list)):
        if word_list[i].isdigit():
            index = i
            break
    return ' '.join(word_list[1:index + 1])


def parse(path):
    try:
        with open(path, encoding="utf8") as fp:
            info = dict()  # Create an empty dictionary
            soup = BeautifulSoup(fp, 'html.parser')  # Load the html file parser using BeautifulSoup
            info['bookTitle'] = clean_str(soup.find('h1', id='bookTitle').text)  # Find the book title
            info['bookSeries'] = clean_str(soup.find('h2', id='bookSeries').text)[1:-1]  # Find the book series
            info['bookAuthors'] = [clean_str(author.text) for author in
                                   soup.find_all('div', 'authorName__container')]  # Find the book authors
            info['ratingValue'] = clean_str(soup.find('span', itemprop='ratingValue').text)  # Find the rating value
            info['ratingCount'] = soup.find('a', 'gr-hyperlink', href='#other_reviews').text.split()[
                0]  # Find the rating count
            info['reviewCount'] = soup.find_all('a', 'gr-hyperlink', href='#other_reviews')[1].text.split()[
                0]  # Find the review count
            plot = soup.find('div', id='description')  # Find the plot
            if plot:  # If the plot exists
                info['plot'] = clean_str(plot.find_all('span')[-1].text)  # Store the information in the dictionary
            else:
                info['plot'] = ''  # Otherwise store an empty string
            numberOfPages = soup.find('span', itemprop='numberOfPages')  # Find the number of pages
            if numberOfPages:  # If it exists
                info['numberOfPages'] = numberOfPages.text.split()[0]  # Save the number
            else:
                info['numberOfPages'] = ''  # Otherwise save an empty string
            published = soup.find_all('div', 'row')  # Find the publication date
            if len(soup.find_all('div', 'row')) > 1:  # If it exists
                first_published = published[1].find('nobr')  # Try to find the first publication date
                if first_published:  # If the first publication date exists
                    # Example of first_published.text  = (first published June 21st 2003)
                    info['published'] = first_published.text.strip()[17:-1]  # Save it in the dictionary
                else:
                    info['published'] = get_publication_date(published[1].text)  # Otherwise save the publication date
            else:
                info['published'] = ''  # If the publication date doesn't exist save an empty string
            characters = soup.find('div', 'infoBoxRowTitle', string=re.compile("Characters"))  # Find the characters
            if characters:  # If they exist
                info['characters'] = clean_str(re.sub('...(less|more)', '',
                                                      characters.find_next_sibling('div').text)).split(
                    ', ')  # Save the list containing the characters in the dictionary
            else:
                info['characters'] = []  # Otherwise save an empty list
            setting = soup.find('div', 'infoBoxRowTitle', string=re.compile("Setting"))  # Find the settings
            if not setting:  # If they don't exists
                info['setting'] = []  # Save an empty list
            else:
                setting = setting.find_next_sibling('div').find_all('a')
                info['setting'] = [clean_str(s.text) for s in
                                   setting]  # Otherwise save the list containing the settings
            info['url'] = soup.find('meta', property='og:url')['content']  # Find the page url
            return info  # Return the dictionary
    except IOError:  # If the file is missing
        return None  # Return None


def from_html_to_tsv(html_path, tsv_path):
    with open(tsv_path, mode='w', encoding="utf8") as tsv_file:
        if not os.path.isfile(html_path):  # Check if the file exists
            tsv_file.close()
            return
        html_info = parse(html_path)  # Parse the html file and save the result in the html_info dictionary
        try:
            if html_info['plot'] and (detect(html_info['plot']) != 'en'):  # If the plot language is not english
                tsv_file.close()  # Close the tvs file
                return
        except:
            print('Exception', html_info)
            tsv_file.close()
            return

        # Otherwise print the keys on file
        tsv_file.write('\t'.join(html_info.keys()))
        tsv_file.write('\n')

        # Then print the values on file
        for k in html_info:
            if k in ['bookAuthors', 'characters', 'setting']:
                tsv_file.write(from_list_to_str(html_info[k]) + '\t')
            elif k == 'url':
                tsv_file.write(html_info[k])
            else:
                tsv_file.write(html_info[k] + '\t')


def process_pages(starting_page, ending_page, page_folder='./data/', destination_folder='./data/tsv/',
                  html_prefix='book', tsv_prefix='article', n_book_per_page=100, starting_number=1,
                  print_page=False, print_book=False):
    # For each page in the interval
    for num_page in range(starting_page, ending_page + 1):
        if print_page:  # If required print the page number
            print('Page:', num_page)
        # For each book in the page
        for i in range(n_book_per_page):
            if print_book:  # If required print the book number
                print('Book:', i)
            idx = str((num_page - 1) * n_book_per_page + i + starting_number)
            # Find the html path and the tsv path using the page and the book number information
            html_path = page_folder + 'page_' + str(num_page) + '/' + html_prefix + '_' + idx + '.html'
            tsv_path = destination_folder + tsv_prefix + '_' + idx + '.tsv'
            # Then parse the html_path file and save the results in the tsv_path file
            from_html_to_tsv(html_path, tsv_path)


if __name__ == '__main__':
    url_request()
    download_pages()
    process_pages(1, 1, n_book_per_page=30000, starting_number=0, tsv_prefix='book', print_book=True)
