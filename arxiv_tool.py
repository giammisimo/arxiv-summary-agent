import requests
import feedparser
from io import BytesIO
from pdfminer.high_level import extract_text
import re

def prune_paper(text: str) -> str:
    """
    Rimuove le sezioni del paper non utili al riassunto. 
    """
    a = (len(text))

    text = re.sub(r'( \n){3,}','',text)
    text = re.sub(r'\n.\n','',text)
    text = re.sub(r'\s\s\s','',text)

    match = re.search(r"\s+References", text, re.IGNORECASE)
    if match:
        text = text[:match.start()]

    match = re.search(f'\n\\(?[0-9.]+\\)?[.\\-\\s]+Conclusion', text, re.IGNORECASE)
    if match:
        text = text[:match.start()]

    print(a, len(text))
    return text

def get_links(paper_id: str) -> dict:
    """
    Ottiene i link associati ad un paper Arxiv tramite la rispettiva API.

    Args:
        `paper_id` (`str`): L'ID associato al paper su Arxiv

    Returns:
        `dict`: Dizionario contenente i link associati al paper:
            `arxiv_link`: Link alla pagina arxiv del paper
            `pdf_link`: Link al file pdf contenente il paper
    """
    url = f'http://export.arxiv.org/api/query?id_list={paper_id}'
    data = feedparser.parse(requests.get(url).text)['entries'][0]
    
    result = dict()

    for link in data.links:
        if link.type == "application/pdf":
            result['pdf_link'] = link.href
        elif link.type == "text/html":
            result['arxiv_link'] = link.href

    return result

def get_paper(paper_id: str) -> dict:
    """
    Ottiene i link e il contenuto testuale del paper Arxiv richiesto.
    Args:
        `paper_id` (`str`): L'ID associato al paper su Arxiv

    Returns:
        `dict`: Dizionario contenente i dati associati al paper:
            `arxiv_link`: Link alla pagina arxiv del paper
            `pdf_link`: Link al file pdf contenente il paper
            `text`: Contenuto testuale del paper
    """
    print(f'REQUESTED PAPER {paper_id}')

    try:
        paper = dict()
        paper['arxiv_link'] = f'https://arxiv.org/abs/{paper_id}'
        paper['text'] = open(f'temp/{paper_id}.txt','r').read()
    except:
        print(f'DOWNLOADING {paper_id}')    
        paper = get_links(paper_id)
        pdf = requests.get(paper['pdf_link'])
        pdf_content = BytesIO(pdf.content)

        paper['text'] = prune_paper(extract_text(pdf_content))

        with open(f'temp/{paper_id}.txt','w') as f:
            f.write(paper['text'])

    return paper