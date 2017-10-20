

# standard libraries
import time
import subprocess

# nonstandard libraries
import pyscreenshot as ImageGrab
import pyautogui

# Drag distance 

#im=ImageGrab.grab(bbox=(10,10,510,510)):


repeats = 960 
distance = 1

print 'Top left:'
for i in xrange(3):
    print 'Starting iteration in {}...'.format(3-i)
    time.sleep(1.0)
top = pyautogui.position()
print 'Top:',top

print 'Bottom right:'
for i in xrange(3):
    print 'Starting iteration in {}...'.format(3-i)
    time.sleep(1.0)

bottom = pyautogui.position()

print 'Bottom:',bottom

x_diff,y_diff = bottom[0] - top[0],bottom[1] - top[1]
bottom_adjust = (top[0] + 16*(x_diff/16),top[1] + 16*(y_diff/16))

print 'Saved bottom:',bottom_adjust

print 'Default:'
for i in xrange(3):
    print 'Starting iteration in {}...'.format(3-i)
    time.sleep(1.0)
default = pyautogui.position()
print 'Default:',default

for r in xrange(repeats):
    print 'Picture {}...'.format(r)
    pyautogui.moveTo(*default)
    pyautogui.dragRel(-distance, -distance, duration=0.4)
    pyautogui.moveTo(1,1)
    
    time.sleep(0.5)
    im = ImageGrab.grab(bbox=(top[0],top[1],bottom_adjust[0],bottom_adjust[1]))
    im.save('./video/test_{}.png'.format('%05d' % r))    

#subprocess.call('ffmpeg -f image2 -r 1/24 -i video/A12_%05d.png -vcodec mpeg4 -y A12.mp4')

