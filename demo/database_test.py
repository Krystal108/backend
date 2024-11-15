import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_id",
    port="3307"
)

mycursor = mydb.cursor()

# Reset auto increment to 1
mycursor.execute("ALTER TABLE register AUTO_INCREMENT = 1")




