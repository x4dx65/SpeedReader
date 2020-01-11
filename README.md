## Update

For the latest version and the dataset visit: https://github.com/brave/speedreader-paper-materials

## Dataset Structure

In case of using the `preprocess` method for preparing training data, the dataset must be in the following format:
```
Dataset/
  |- D1
     |-wired.com
        |-init-html.html
        |-url.txt
     |-nytimes.com
     \...
  |- D2
  |- D3 
```
- Labels of the URLs with their corresponding folder (D1,D2,D3) are available in `labels.csv`
- URLs in `labels.csv` should be replaced with their folder name (e.g. `wired.com`)
- Each folder should have `url.txt` which contains directory names and the url in `JSON` format:
    - `{"url": "https://www.wired.com/story/...", "dir": "wired.com"}`
 

## Getting Performance Result

- Run `pip install --upgrade -r requirements.txt` to get all dependencies.
- Note: change the arguments based on your system and either you want the final DOM results or init HTML
- Run `python model.py --datapath "/path/to/Dataset/" --labels "/path/to/labels.csv" --filetype "init-html.html" --cmodel "/name/of/the/output/model.c"`
- After it's done, the classification result will be printed on the screen
- You can ignore the `temp_labels_*.csv` file

## Getting C Model

- Replace the third step of previous section with `python model.py --datapath "/path/to/Dataset/" --labels "/path/to/labels.csv" --filetype "init-html.html" --cmodel "/name/of/the/output/model.c"`
- The C model file will be available in `/name/of/the/output/model.c`

Tested on Ubuntu 16.04.5 LTS

## Note

- Everything in this repository is written in Python 2.7

