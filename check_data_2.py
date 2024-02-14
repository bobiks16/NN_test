def check_labels_content(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            with open(folder + filename, "r") as label_file:
                label_content = label_file.readline().split()

                label_content[0] = int(label_content[0])
                label_content[1:] = list(map(float, label_content[1:]))

                if label_content[0] not in range(29):
                    print(f"\n{folder + filename}\n\t- Invalid class index")
                
                for coordinate in label_content[1:]:
                    if coordinate > 1 or coordinate < 0:
                        print(f"\n{folder + filename}\n\t- Invalid coordinates")


img_array = np.expand_dims(img_array, axis=0)

class_prediction, bbox_prediction = model.predict(img_array)
class_prediction = np.argmax(class_prediction, axis=1)[0]
bbox_prediction = bbox_prediction[0]