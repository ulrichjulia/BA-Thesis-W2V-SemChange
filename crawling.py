from bs4 import BeautifulSoup
import requests


def grab_redtype_links(url):
    """
    Using the WebArchive TimeMachine interface, the specified url
        (here for Reddit and TypePad) is filtered for any linked content.
        """
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    alllinks = []
    definiteli = []
    alllinks.append(url)

    for link in soup.find_all('a'):
        li = link.get('href')
        alllinks.append(li)

    for item in alllinks:
        try:
            if item.startswith('/web/2'):
                newitem = 'https://web.archive.org'+item
                definiteli.append(newitem)

            elif item.startswith('https://'):
                definiteli.append(item)
        except AttributeError:
            continue

    return definiteli


def crawl_red_list(links, name, date):
    """
    The links which were collected with grab_redtype_links() are
        accessed one after another and the text which is found on these websites is extracted and stored to a file.
        """
    linkz = grab_redtype_links(links)
    for link in linkz:
        try:
            source = requests.get(str(link)).text

            soup = BeautifulSoup(source, 'lxml')

            with open('{}_{}.txt'.format(name, date), 'a') as fi:
                fi.write(soup.get_text())
        except requests.exceptions.TooManyRedirects:
            continue