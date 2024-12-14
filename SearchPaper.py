import requests
import time


def search_paper_with_retries(title, fields=None, retries=3, backoff_factor=1):
    """
    Search for a paper by title using Semantic Scholar API with retry mechanism.

    Args:
        title (str): The title of the paper to search for.
        fields (str, optional): A comma-separated list of fields to include in the response.
        retries (int): Number of retry attempts in case of timeout or failure.
        backoff_factor (int): Factor to increase delay between retries.

    Returns:
        dict or str: Returns paper details as a dictionary if found, or an error message.
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {"query": title}
    if fields:
        params["fields"] = fields

    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return "Title match not found."
            else:
                print(
                    f"Attempt {attempt + 1}: Received status code {response.status_code} with message {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error - {str(e)}")

        # Wait before retrying
        time.sleep(backoff_factor * (2 ** attempt))

    return "Failed to retrieve paper after multiple attempts."


# Example usage
if __name__ == "__main__":
    paper_title = "Object Tracking Benchmark"
    requested_fields = "title,authors"

    # Use the function with retries
    result = search_paper_with_retries(paper_title, fields=requested_fields, retries=5, backoff_factor=2)
    print(result)


# {'data': [{'paperId': 'd2c733e34d48784a37d717fe43d9e93277a8c53e',
#            'title': 'ImageNet: A large-scale hierarchical image database',
#            'authors': [{'authorId': '153302678', 'name': 'Jia Deng'},
#                        {'authorId': '144847596', 'name': 'Wei Dong'},
#                        {'authorId': '2166511', 'name': 'R. Socher'},
#                        {'authorId': '2040091191', 'name': 'Li-Jia Li'},
#                        {'authorId': '94451829', 'name': 'K. Li'},
#                        {'authorId': '48004138', 'name': 'Li Fei-Fei'}],
#            'matchScore': 190.07565}]}

