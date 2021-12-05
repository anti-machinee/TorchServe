#curl -X OPTIONS http://localhost:8080
curl http://0.0.0.0:8080/ping
curl -X POST http://0.0.0.0:8080/predictions/demo -F "image=@test.jpg"
