class ScraperMerger:
    def __init__(self, scrapers: list):
        self.scrapers = scrapers

    def get_merged_payload(self, query: str):
        """
        Calls get_payload on all scrapers and merges results.
        If keys overlap, the later scraper's data will overwrite the earlier one.
        """
        master_dict = {}

        print("Merging Files using complex data storing algorithms...")
        for scraper in self.scrapers:
            print(f"[*] Running scraper: {scraper.__class__.__name__}")
            # Call each scraper's individual method
            data = scraper.get_payload(query)

            # Merge into the master dictionary using the Pipe operator
            # This creates a new dictionary containing all keys from both
            master_dict |= data  # cite: 1.1, 1.3

        return master_dict