import os
import argparse
import requests
import psycopg2
import json
from threading import Thread
from src.config import hostname, password, username, database


def photo_scrape(directory, n):
    # make the dir first
    if not os.path.isdir(directory):
        os.mkdir(directory)

    links = query_links(n)
    thread_list = []

    # start threads
    for index in range(n):
        th = Thread(target=save_image, args=(links[index], directory))
        thread_list.append(th)
        th.start()

    # join
    for th in thread_list:
        th.join()


# download and save a given image
def save_image(link, directory):
    print("Attempting to download: " + link)
    r = requests.get(link)
    if r.status_code == 200:
        image_id = link.split('/')[-1]
        filename = os.sep.join([directory, image_id])
        print(f'Saving to filename: {filename}')
        with open(filename, 'wb') as f:
            f.write(r.content)
    else:
        print("Couldn't download from link: " + link)


def query_links(n):
    pgsql = database_connect()
    cur = pgsql.cursor()
    cur.execute("""SELECT photos 
                    FROM ads
                    WHERE status='open' 
                        AND ads_type='df'
                    ORDER BY random()"""
                f"LIMIT {n}")
    photos = []
    for photo_json in cur.fetchall():
        if isinstance(photo_json[0], str):
            photo_arr = json.loads(photo_json[0])
        else:
            photo_arr = []

        for photo in photo_arr:
            photos.append(photo['img'])

    pgsql.close()

    return photos


def database_connect():
    return psycopg2.connect(host=hostname,
                            user=username,
                            password=password,
                            dbname=database)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape from stock images')
    parser.add_argument('-f', dest='folder', help='Specify the folder where to place the images.')
    parser.add_argument('-u', dest='url', help='Specify the place from where to scrape.')
    args = parser.parse_args()

    # if args.url is None:
    #     parser.print_help()
    #     sys.exit(0)
    # else:
    # define the folder
    if args.folder is None:
        directory = "."
    else:
        directory = args.folder

    photo_scrape(directory, 10)

    print("Done.")
