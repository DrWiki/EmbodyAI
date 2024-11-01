from DM_CAN import *
import serial
import time

Motor1 = Motor(DM_Motor_Type.DM4340, 0x08, 0x1C)
Motor2 = Motor(DM_Motor_Type.DM4340, 0x09, 0x1D)
Motor3 = Motor(DM_Motor_Type.DM4340, 0x0A, 0x1E)
Motor4 = Motor(DM_Motor_Type.DM4310, 0x0B, 0x1F)
Motor5 = Motor(DM_Motor_Type.DM4310, 0x0C, 0x20)
Motor6 = Motor(DM_Motor_Type.DM4310, 0x0D, 0x21)
Motor7 = Motor(DM_Motor_Type.DM4310, 0x0E, 0x22)

serial_device = serial.Serial('COM66', 921600, timeout=0.5)

Motor_Control = MotorControl(serial_device)
Motor_Control.addMotor(Motor1)
Motor_Control.addMotor(Motor2)
Motor_Control.addMotor(Motor3)
Motor_Control.addMotor(Motor4)
Motor_Control.addMotor(Motor5)
Motor_Control.addMotor(Motor6)
Motor_Control.addMotor(Motor7)

Motor_Control.zero_position(Motor1)
time.sleep(0.2)
Motor_Control.zero_position(Motor2)
time.sleep(0.2)
Motor_Control.zero_position(Motor3)
time.sleep(0.2)
Motor_Control.zero_position(Motor4)
time.sleep(0.2)
Motor_Control.zero_position(Motor5)
time.sleep(0.2)
Motor_Control.zero_position(Motor6)
time.sleep(0.2)
Motor_Control.zero_position(Motor7)
time.sleep(0.2)

print("Set Zero Position Successfully!")

# 语句结束关闭串口
serial_device.close()

