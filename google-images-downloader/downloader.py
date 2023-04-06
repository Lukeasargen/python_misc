
import os
import json
import time
import logging
import urllib.request
import urllib.error
from urllib.parse import urlparse, quote

from multiprocessing import Pool
from user_agent import generate_user_agent
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def get_image_links(main_keyword, supplemented_keywords, link_file_path, num_requested=10):
    """get image links with selenium
    
    Args:
        main_keyword (str): main keyword
        supplemented_keywords (list[str]): list of supplemented keywords
        link_file_path (str): path of the file to store the links
        num_requested (int, optional): maximum number of images to download
    
    Returns:
        None
    """
    print("link_file_path :", link_file_path)
    print("num_requested :", num_requested)

    img_urls = set()
    driver = webdriver.Firefox(executable_path='geckodriver.exe')
    for i in range(len(supplemented_keywords)):
        search_query = quote(main_keyword + ' ' + supplemented_keywords[i])
        url = "https://www.google.com/search?q="+search_query+"&source=lnms&tbm=isch"
        print("url :", url)
        driver.get(url)
        start_time = time.time()
        while time.time()-start_time < 10:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)  # wait for page to load

            # find all the picture and count them
            # imges = driver.find_elements_by_xpath('//div[@class="rg_meta"]') # not working anymore
            # imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]') # not working anymore
            thumbs = driver.find_elements_by_xpath('//a[@class="wXeWr islib nfEiy mM5pbd"]')
            print("images so far :", len(thumbs))


            # check for the more pictures button
            try:
                driver.find_element_by_xpath("//input[@value='Show more results']").click()
            except Exception as e:
                print("Process-{0} reach the end of page or get the maximum number of requested images".format(main_keyword))

            if len(thumbs) > num_requested:
                break

        max_count = min(num_requested, len(thumbs))

        print("max_count :", max_count)

        for j in range(max_count):
            thumb = thumbs[j]
            try:
                thumb.click()
                time.sleep(1)
            except e:
                print("Error clicking one thumbnail")

            url_elements = driver.find_elements_by_xpath('//img[@class="n3VNCb"]')
            for url_element in url_elements:
                try:
                    url = url_element.get_attribute('src')
                except e:
                    print("Error getting one url")

                if url.startswith('http') and not url.startswith('https://encrypted-tbn0.gstatic.com'):
                    img_urls.add(url)
                    print("Found image url: " + url)

        print('Process-{} add keyword {} , got {} image urls so far'.format(main_keyword, supplemented_keywords[i], len(img_urls)))

    # END supplemented loop


    print('Process-{} totally get {} images'.format(main_keyword, len(img_urls)))
    driver.quit()


    with open(link_file_path, 'w') as wf:
        for url in img_urls:
            wf.write(url +'\n')
    print('Store all the links in file {0}'.format(link_file_path))


def download_images(link_file_path, download_dir, log_dir):
    """download images whose links are in the link file
    
    Args:
        link_file_path (str): path of file containing links of images
        download_dir (str): directory to store the downloaded images
    
    Returns:
        None
    """
    print('Start downloading with link file {0}..........'.format(link_file_path))
    print("log_dir :", log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    main_keyword = link_file_path.split('/')[-1]
    log_file = log_dir + 'download_selenium_{0}.log'.format(main_keyword)
    print("log_file :", log_file)
    logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")
    img_dir = download_dir + main_keyword + '/'
    print("img_dir :", img_dir)
    count = 0
    headers = {}
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # start to download images

    print("link_file_path :", link_file_path)

    with open(link_file_path, 'r') as rf:
        for link in rf:
            try:
                o = urlparse(link)
                ref = o.scheme + '://' + o.hostname
                #ref = 'https://www.google.com'
                ua = generate_user_agent()
                headers['User-Agent'] = ua
                headers['referer'] = ref
                print('\n{0}\n{1}\n{2}'.format(link.strip(), ref, ua))
                req = urllib.request.Request(link.strip(), headers = headers)
                response = urllib.request.urlopen(req)
                data = response.read()
                print("link :", link)
                file_path = img_dir + '{0}.jpg'.format(count)
                with open(file_path, 'wb') as wf:
                    wf.write(data)
                print('Process-"{0}" download image "{1}/{2}.jpg"'.format(main_keyword, main_keyword, count))
                count += 1

            except Exception as e:
                print('Unexpected Error')
                logging.error('Unexpeted error while downloading image {0}error type:{1}, args:{2}'.format(link, type(e), e.args))
                continue


if __name__ == "__main__":
    main_keywords = [
                    'romano lagotto white',
                    'romano lagotto white italian',
                    'romano lagotto small',
                    'romano lagotto puppies',
                    ]

    supplemented_keywords = [
                "",
                ]

    num = 1000


    download_dir = './data/'
    link_files_dir = './data/link_files/'
    log_dir = './logs/'
    for d in [download_dir, link_files_dir, log_dir]:
        print(d)
        if not os.path.exists(d):
            os.makedirs(d)

    ###################################
    # get image links and store in file
    ###################################
    # single process
    # for keyword in main_keywords:
    #     link_file_path = link_files_dir + keyword
    #     get_image_links(keyword, supplemented_keywords, link_file_path, num_requested=num)
    

    # multiple processes

    n_cores = 4

    p = Pool(n_cores) # default number of process is the number of cores of your CPU, change it by yourself
    for keyword in main_keywords:
        p.apply_async(get_image_links, args=(keyword, supplemented_keywords, link_files_dir+keyword, num))
    p.close()
    p.join()
    print('Finsh getting all image links')
    
    ###################################
    # download images with link file
    ###################################
    # single process
    # for keyword in main_keywords:
    #     link_file_path = link_files_dir + keyword
    #     download_images(link_file_path, download_dir, log_dir)
    
    # multiple processes
    p = Pool(n_cores) # default number of process is the number of cores of your CPU, change it by yourself
    for keyword in main_keywords:
        p.apply_async(download_images, args=(link_files_dir+keyword, download_dir, log_dir))
    p.close()
    p.join()
    print('Finish downloading all images')