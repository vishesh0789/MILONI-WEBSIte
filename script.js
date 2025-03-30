document.querySelector('#login form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from submitting
    var username = document.querySelector('#username').value;
    var password = document.querySelector('#password').value;
    alert('Login attempted for ' + username + ' with password ' + password);
});
