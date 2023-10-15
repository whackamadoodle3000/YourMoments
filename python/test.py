#pip3 install lxml, for beautifulsoup html parser
from bensound import BensoundAPI

api = BensoundAPI()
api.extract_all_data()


#list = api.get_channel_list()

#electronic_list = api.get_songs_by_channel('Electronica')

#print(list)