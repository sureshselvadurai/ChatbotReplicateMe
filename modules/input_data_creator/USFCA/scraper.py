from bs4 import BeautifulSoup


def print_all_p_elements(html_file):
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
        p_elements = soup.find_all('p')
        for p in p_elements:
            print("------------")
            print(p.get_text())

def print_elements_by_class(html_file, tag_name, class_name):
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
        elements = soup.find_all(tag_name, class_=class_name)
        for element in elements:
            print(element.get_text())

if __name__ == '__main__':
    html_file = '/Users/sureshrajaselvadurai/PycharmProjects/Coursework/Chatbot Replicate.me/input/USFCA/html/student_employment.html'

    tag_name = 'h2'
    class_name = 'accordion-title-h'
    # print_elements_by_class(html_file, tag_name, class_name)
    print_all_p_elements(html_file)