from flask import Flask, request, jsonify


app=Flask(__name__)

@app.route('/chat', method=["GET", "POST"])
def chatbot():
    chatInput=request.form["chatInput"]
    return jsonify(chatbotreply=chatwithbot(chatInput))



if __name__=="__main__":
    app.run(debug=True)