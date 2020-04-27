# Covid-19-Data-Analysis
This project helps people understand the effects of Covid-19 on our society.

The program we created parses through article titles and abstracts that focus on the topic of COVID-19.
The reason for parsing through these instead of the whole article is because these give a concise short overview of the topic and there are possibilities for articles to diverge from the main focus.
The frequency of occurrence of each word is kept tracked and the most common words from the collective titles and abstracts are represented in a bar graph comparing the frequencies.
The least common words vary and appear in a list form with the lowest frequency words; majority occurring only once out of a large pool of articles.
From the data the algorithm gives, the most common words can give clarification on what information can be considered as a more “reliable source” compared to the articles with terms that are not as frequent.
The method used for parsing utilizes a library called sklearn in which its algorithm learns the relationships between each word as it goes through and collects the ones that it deems to be correlated with the topic.