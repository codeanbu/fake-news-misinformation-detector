import trafilatura


def extract_article(url):
    """
    Downloads and extracts the main content from a news article.
    """

    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        raise Exception("Unable to download the webpage.")

    text = trafilatura.extract(downloaded)

    if text is None:
        raise Exception("Unable to extract article text.")

    return {
        "title": url,
        "text": text
    }