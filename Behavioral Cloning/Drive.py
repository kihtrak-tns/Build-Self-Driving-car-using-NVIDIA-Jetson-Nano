#import socketio

#import eventlet
#from flask import Flask
from keras.models import load_model

# Web sockets provide communication between the client and the server. 
## In our case, we're going to establish a bidrectional communication with the simulator using socketio
sio = socketio.Server()
app = Flask(__name__)									# Flask instance for web app



@sio.on('telemetry')
# def telemetry(sid, data):
	
# When there's a connection with the client, we want to have an event handler
@sio.on('connect')
def connect(sid, environ) :
	print('Connected')
	send_control(0, 0)

def send_control(steering_angle, throttle):
	# Events to the Udacity Simulator
	sio.emit('steer', data = {
		'steering_angle': steering_angle.__str__(), 
		'throttle': throttle.__str__()
		})
	
if __name__ == '__main__' :
	model = load_model('model.h5')
	# Middleware dispatches traffic to socketio web server
	app = socketio.Middleware(sio, app)
	# We make a gateway between the client and the server
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
	