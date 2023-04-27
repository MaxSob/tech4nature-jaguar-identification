from django.core.files.storage import FileSystemStorage


def save_image_static(file):
    
    fs = FileSystemStorage()
    filename = fs.save(file.name, file)
    file_path = fs.url(filename)

    return file_path