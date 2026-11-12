const e = new URLSearchParams(location.search).get("e");
if (e) {
  const el = document.getElementById("err");
  el.textContent = e === "rate" ? "Too many attempts. Wait a minute and try again." : "Wrong username or password.";
  el.classList.remove("hidden");
}
