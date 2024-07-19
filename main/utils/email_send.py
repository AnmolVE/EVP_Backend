from django.core.mail import send_mail
from django.conf import settings

def send_email_to_users(emails):
    subject = "This email is from Django Server"
    message = "This is a text message from Django Server Email"
    from_email = settings.EMAIL_HOST_USER
    recipient_list = emails
    send_mail(subject, message, from_email, recipient_list)
