# -*- coding: UTF-8 -*-
import urllib
import urllib2
import requests
import re
from bs4 import BeautifulSoup as bsp 
import sys
import time
from urllib import quote


def get_all_url(url_home):
    headers = {'User-agent' : 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:22.0) Gecko/20100101 Firefox/22.0'}
    request = urllib2.Request(url_home, headers = headers)
    web = urllib2.urlopen(request)
    urls = []
    soup =bsp(web.read(), 'lxml')
    tags_a =soup.findAll(name='a',attrs={'href':re.compile("^/photo/show/")})
    try :
       for tag_a in tags_a:
           urls.append(tag_a['href'])
    except:
       pass

    urls = list(set(urls))
    return urls

def extract_info(url):
    post_id = re.findall("\d+",url)[0]
	
    url = 'http://www.chictopia.com'+url
	
    html = urllib2.urlopen(url)
    soup = bsp(html, "lxml")

    out = soup.find(id="image_wrap")
    imgUrl = out.img['src']
    path = 'img/'
    img_path = path+post_id+'.jpg'
    urllib.urlretrieve(imgUrl.encode('utf-8'), img_path)

    name_div = soup.find(id="name_div")
    username = name_div.a.text
    data_path = 'after_10000_'
    #data_path = '/ais/gobi4/fashion/chictopia/'
    with open(data_path+'user_post_photo.txt', 'a') as w_f:
	    w_f.write(username+'\t'+post_id+'\t'+img_path+'\t'+ imgUrl.encode('utf-8')+'\n')
    w_f.close()

    loc_div = soup.find(id="loc_div")
    if loc_div.a:
	    loc = loc_div.a.text
    else:
	    loc = 'earth'
	
    if soup.select("div.av_info > div.px10"):
        div = soup.select("div.av_info > div.px10")[0]
        point_str = div.text
    else:
    	point_str = '0 chic points'

    point = point_str.split()
	#print point[0]

    fans = soup.find(id="fan_count").text
    fans = fans.split()
    fan_cnt = fans[0]
	#print fan_cnt

    with open(data_path+'user_info.txt','a') as w_f:
	    w_f.write(username+'\t'+loc+'\t'+point[0]+'\t'+fan_cnt+'\n')
    w_f.close()

    vote = soup.find(id=re.compile(r"^vote_text")).text
    vote = vote.split()
    vote_cnt = vote[0]

    favorites = soup.find(id=re.compile(r"^favorite_text")).text
    favorites = favorites.split()
    favorite_cnt = favorites[0]
	#print favorite_cnt

    comments = soup.find(id=re.compile(r"^comment_text")).text
    comments = comments.split()
    comments_cnt = comments[0]
	#print comments_cnt

    with open(data_path+'cnt.txt', 'a') as w_f:
	    w_f.write(post_id+'\t'+vote_cnt+'\t'+favorite_cnt+'\t'+comments_cnt+'\n')
    w_f.close()

    tags = soup.select("div#tag_boxes > div.left > div.px10")
    str_tag = ''
    for tag in tags:
	    str_tag = str_tag + '\t'+tag.text

    with open(data_path+'tags.txt', 'a') as w_f:
	    w_f.write(post_id+str_tag+'\n')
    w_f.close()

	
    garments = soup.find_all("div", class_=["garmentLinks"])
	#print garments


    for garment in garments:
	    color = garment.findChildren(href=re.compile(r".*color.*"))
	    item = garment.findChildren(id=re.compile(r"^(garment).*\d$"))
   
	    color_str = ''
	    item_str = ''
	    if len(color):
	    	color_str = color_str + '\t' + color[0].text

	    if len(item):
	    	item_str = item_str + '\t' + item[0].text
	    
	    with open(data_path+'color_item.txt', 'a') as w_f:
	    	w_f.write(post_id+color_str+item_str+'\n')
	    w_f.close()

    title = soup.find("h1", class_=["photo_title"])
    if title:
    	title = title.text
    else:
    	title = 'no-title'
	#print title

    update_time = soup.select("div#title_bar > div.left > div.px10")
    if len(update_time):
        update_time = update_time[0].text
        update_time = update_time.split()
        update_time = update_time[2]+'\t'+update_time[3]+'\t'+update_time[4]
    else:
    	update_time = 'no-time'
	#print update_time

    with open(data_path+'post_info.txt', 'a') as w_f:
	    w_f.write(post_id+'\t'+title+'\t'+update_time+'\n')
    w_f.close()


    comments = soup.select("div.single_comment > div.left > a > img.hoverDarken")
	# print "comments"
	# print comments
	# if len(comments):
	#     print comments[0]['alt']#user name for the comments

    user_comments = soup.select("div.single_comment > div.userComment > div > meta")
	# if len(user_comments):
	#     print user_comments[0]['content']

    comments_contents = soup.select("div.single_comment > div.userComment > div.comment_content")
	# if len(comments_contents):
	#     print comments_contents[0].text
    len_comments = len(comments)
    for i in xrange(len_comments):
	    with open(data_path+'comments.txt', 'a') as w_f:
		    w_f.write(post_id+'\t'+comments[i]['alt']+'\t'+user_comments[i]['content']+'\t'+comments_contents[i].text+'\n')
	    w_f.close()


sys.getdefaultencoding()
reload(sys)
sys.setdefaultencoding('utf-8')
root_url = 'http://www.chictopia.com/browse/people/'


for i in xrange(18203):
    time.sleep(2)
    start_time = time.time()
    lurls = get_all_url(root_url+str(i+10506))
    for ret in lurls:
        time.sleep(1)
    	extract_info(ret)
        #print ret
    end_time = time.time()
    print "page i:"
    print i+1
    print end_time - start_time




