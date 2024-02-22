import pika, sys, os

def main():
    # wait for imput from terminal
    # send message to rabbitMQ

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='test')

    while True:
        message = input("Enter message: ")
        if message == 'exit':
            break
        channel.basic_publish(exchange='', routing_key='test', body=message)
        print(f" [x] Sent {message}")

    connection.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)