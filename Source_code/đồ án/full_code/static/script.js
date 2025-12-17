document.getElementById("dirPicker").addEventListener("change", async (event) => {
  const files = event.target.files;
  const formData = new FormData();
  for (let file of files) {
    formData.append("files[]", file);
  }

  const response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  const fileList = document.getElementById("fileList");
  fileList.innerHTML = "";

  data.forEach(result => {
    const li = document.createElement("li");
    li.textContent = `${result.filename} → ${result.status} (P=${result.probability}) | ${JSON.stringify(result.counts)}`;
    fileList.appendChild(li);
  });

  document.getElementById("serverResult").style.display = "block";
  document.getElementById("serverResult").textContent = "✅ Đã xử lý xong " + data.length + " ảnh.";
});