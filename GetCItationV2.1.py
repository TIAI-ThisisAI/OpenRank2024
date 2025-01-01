import requests
import csv
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def get_paper_details(paper_id):
    """
    Get details of a paper including its citation count.

    Args:
        paper_id (str): The paperId to get details for.

    Returns:
        dict: A dictionary containing paper details including citation count.
    """
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": "citationCount"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Status code {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return {}


def fetch_citations(paper_id, fields=None, limit=1000, start_offset=0):
    """
    Fetch all citations for a given paper using Semantic Scholar API.

    Args:
        paper_id (str): The paperId for which to fetch citations.
        fields (str, optional): A comma-separated list of fields to include in the response.
        limit (int): Maximum number of citations per API request (default is 1000).
        start_offset (int): The offset to start fetching from.

    Returns:
        list: A list of dictionaries containing citation information.
    """
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    offset = start_offset

    # Set up retries to handle connection issues
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504, 429])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    while True:
        params = {"offset": offset, "limit": limit}
        if fields:
            params["fields"] = fields

        try:
            response = session.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                citations = data.get("data", [])
                yield citations

                # Update counters
                offset += len(citations)

                # Check if there are more results
                if len(citations) < limit:
                    break
            else:
                print(f"Error: Status code {response.status_code} - {response.text}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            break

        # Delay to avoid rate-limiting
        time.sleep(1)


def save_citations_to_csv(citations, filename):
    """
    Save citation information to a CSV file.

    Args:
        citations (list): A list of citation data.
        filename (str): The name of the CSV file to save.
    """
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header if the file is empty
        if file.tell() == 0:
            writer.writerow(["Paper Title", "Authors", "Publication Year"])

        for citation in citations:
            citing_paper = citation.get("citingPaper", {})
            title = citing_paper.get("title", "Unknown")
            year = citing_paper.get("year", "Unknown")
            authors = ", ".join([author.get("name", "Unknown") for author in citing_paper.get("authors", [])])

            writer.writerow([title, authors, year])


# Main program
if __name__ == "__main__":
    paper_id = "c4c45661501c16064eead6e5d37dcb80d41c7a78"  # Replace with your paperId
    fields = "citingPaper.title,citingPaper.authors,citingPaper.year"
    output_file = "paper/video/mid.csv"

    # Get the total citation count for the paper
    paper_details = get_paper_details(paper_id)
    total_citations = paper_details.get("citationCount", 0)
    print(f"Total citations to fetch: {total_citations}")

    # Fetch citations in a paginated manner and save to CSV
    print("Fetching citations...")
    total_fetched = 0
    start_offset = 0
    while total_fetched < total_citations:
        batch_fetched = 0
        for citations_batch in fetch_citations(paper_id, fields=fields, start_offset=start_offset):
            if not citations_batch:
                break
            print(f"Saving {len(citations_batch)} citations to {output_file}...")
            save_citations_to_csv(citations_batch, output_file)
            total_fetched += len(citations_batch)
            batch_fetched += len(citations_batch)
            start_offset += len(citations_batch)
            if batch_fetched >= 9000:
                print("Reached 9000 citations in this batch, writing to CSV and waiting before continuing...")
                time.sleep(30)  # Wait before continuing to avoid rate limits
                batch_fetched = 0

    print("Done!")
