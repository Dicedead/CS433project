
# file to JSON
 
 
import json
 
 
# the file to be converted
filename = 'data1.txt'
 
dict_track = {}
dict_line = {}
index = -1
line_index =-1
# fields in the sample file
lines = ['line_type','step_number','x','y','z','spawn_in_step','procname']
fields =['track_number', 'track_id', 'parent_id', 'particle', 'number_of_steps', lines]

def create_new(description,  dict_track):
  global index
  index +=1
  dict_track[fields[0]] = index
  dict_track[fields[3]] = description[5]
  dict_track[fields[1]] = description[9]
  dict_track[fields[2]] = description[13]

def read_line(description, dict_line):
  global line_index
  line_index+=1
  dict_tmp= {}
  dict_tmp[lines[1]] = description[0]
  dict_tmp[lines[2]] = description[1]
  dict_tmp[lines[3]] = description[2]
  dict_line[line_index] = dict_tmp
  
  
  
 
with open(filename) as fh:
     
    for line in fh:
        
        # reading line by line from the text file
        description = list( line.replace(',',' ').strip().split(None))
        

        if (description[1] == 'G4Track') : {
           create_new(description,dict_track)
        }
        if(description[1] != 'G4Track' and description[0] != 'Step#' and not(description[0].startswith(':'))) : {
           read_line(description,dict_line)
        }
         
        # for output see below
        print(description)
      
         
      
     
      



out_file = open("test1.json", "w")
json.dump(dict_line, out_file, indent = 4)
out_file.close()
# creating json file       
out_file = open("test2.json", "w")
json.dump(dict_track, out_file, indent = 4)
out_file.close()