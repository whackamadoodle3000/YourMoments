from deepface import DeepFace
import PIL

DEFAULT_FILENAME = "tmp.jpg"

def save_pil(image, filename = DEFAULT_FILENAME):
    image.save(filename)

'''
Takes in a PIL Image and path to database of named reference face images. 
Image names in the database should match names of the people.

Returns name of face if it finds one. 
'''
def id_face(pil_img, db_path):
    save_pil(pil_img)
    dfs = DeepFace.find(img_path = DEFAULT_FILENAME, db_path = db_path)[0]
    if not dfs.shape[0]:
        return None
    return [dfs.sort_values("VGG-Face_cosine")["identity"][i].split("/")[-1][:-4] for i in range(dfs.shape[0])]

'''
Takes in a PIL Image.
Returns top emotion of face if there is a face. If there is no face, returns None.
'''
def id_emotion(pil_img):
    save_pil(pil_img)
    try:
        emotion = DeepFace.analyze(img_path = DEFAULT_FILENAME, actions = ['emotion'])
        return max(emotion[0]['emotion'].keys(), key = lambda k: emotion[0]['emotion'][k])
    except:
        return None # No face

if __name__ == "__main__":
    # id_face()
    im1 = PIL.Image.open("images/lana1.jpg")
    save_pil(im1)  
    print(id_face(im1, "images/"))