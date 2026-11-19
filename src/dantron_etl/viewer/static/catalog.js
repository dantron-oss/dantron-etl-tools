const q = document.getElementById('q');
q.addEventListener('input', () => {
  const term = q.value.trim().toLowerCase();
  for (const card of document.querySelectorAll('.tool')) {
    card.hidden = term && !card.dataset.search.includes(term);
  }
});
