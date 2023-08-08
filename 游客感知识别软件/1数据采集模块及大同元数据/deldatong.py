
#coding=utf-8
import flickrapi
api_key=u'046116555994df41f750b8706de30d5a'
api_secret=u'f3e16800905e9c5d'
flickr=flickrapi.FlickrAPI(api_key,api_secret,cache=True)
fo=open('datacsvdatongraw.csv','w',encoding="gb18030")
ls_first=['photo_id','accuracy','photo_url','datetaken','title','latitude','longitude','tags']
fo.write(",".join(ls_first)+"\n")
try:
     #爬取text为'New York'的照片,这里可以根据自己的需要设置其它的参数
     photos1=flickr.walk(text='datong shanxi',has_geo='1',extras='url_c,date_taken,owner_name,description,tags,geo')
     #photos1=flickr.walk(sort='1',bbox='112.34,39.03,114.33,40.44',content_type='1',has_geo='1',extras='url_c,date_taken,owner_name,description,tags,geo')
     #photos = flickr.photos.search(text='haihe')
except Exception as e:
     print('Error')

for photo in photos1:
     #获得照片的url,设置大小为url_c(具体参数请参看FlickrAPI官方文档介绍)
     ls=[]
     photo_id=photo.get('id')         #Pic id
     title=photo.get('title')     #Pic title
     owner_name=photo.get('ownername')      #Pic owner id
     ls.append(str(photo_id))
     #ls.append(str(owner_name))
     user_id = photo.get('owner')  # Pic owner id
     '''
     people = flickr.people.getInfo(user_id=user_id)
     for p in people:
          num = 0
          for o in p:
               num += 1
               if num > 3:
                    break
               ls.append(str(o.text))
     '''
     #ls.append(str(title))
     ac = photo.get('accuracy')
     ls.append(str(ac))
     photo_url=photo.get('url_c')       #Pic link
     ls.append(str(photo_url))
     datetaken=photo.get('datetaken')
     ls.append(str(datetaken))
     ls.append(str(title))
     la=photo.get('latitude')
     ls.append(str(la))
     lo=photo.get('longitude')
     ls.append(str(lo))
     tags = photo.get('tags')
     ls.append(str(tags))
     fo.write(",".join(ls)+'\n')
fo.close()