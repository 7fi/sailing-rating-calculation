# CrowsNest Backend Logic

This is the backend updating logic for CrowsNest. This project handles the scraping, calculation, and uploading of the data from techscore to the CrowsNest DB.

The latest updates are on the db-format-change branch in preperation for the beta release.

## Usage

main.py is the primary entry point for the new refactored code. main.py currently takes no args. ` config.py contains the conifguration for the code such as:

- Input and output file names
- Techscore accounts to merge
- Should scrape or load from file?
- Should force full re-calculation?
- Should upload to db?
