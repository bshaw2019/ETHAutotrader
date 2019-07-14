import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import arrow
import os.path


def send_email():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    filename = "graphs/" + str(arrow.utcnow().format("YYYY-MM-DD")) + ".png"
    img_data = open(os.path.join(dir_path, filename), 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Graph Generated: ' + filename
    msg['From'] = 'e@mail.cc'
    msg['To'] = 'e@mail.cc'

    text = MIMEText("Your graph has finished being generated.")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(filename))
    msg.attach(image)

    file = open(os.path.join(dir_path, "logs/keys.txt"))
    passwd = file.readline()
    file.close()


    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('laxmaxer@gmail.com', passwd)
    #s.sendmail('Graph Generator', 'rem8aj@virginia.edu', msg.as_string())
    s.sendmail('Graph Generator', 'als5ev@virginia.edu', msg.as_string())
    #s.sendmail("Graph Generator', 'bshaw19@vt.edu', msg.as_string())
    s.quit()
