// Fetch branches from API and render on Leaflet map
(async function(){
    const listEl = document.getElementById('branchList');
    // Initialize map centered on Sri Lanka
    const map = L.map('map').setView([7.8731, 80.7718], 7);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    try{
        const res = await fetch('/api/branches');
        if(!res.ok){
            throw new Error(`HTTP ${res.status} ${res.statusText}`);
        }
        const data = await res.json();
        if(!data.success){
            throw new Error('API returned success=false');
        }

        const branches = data.branches || [];
        if(branches.length === 0){
            throw new Error('No branches returned');
        }

        branches.forEach(b => {
            const marker = L.marker([b.lat, b.lon]).addTo(map);
            marker.bindPopup(`<strong>${b.name}</strong><br>${b.address}`);

            const item = document.createElement('div');
            item.className = 'branch-item';
            item.innerHTML = `<strong>${b.name}</strong><div>${b.address}</div>`;
            item.addEventListener('click', () => {
                map.setView([b.lat, b.lon], 13);
                marker.openPopup();
            });
            listEl.appendChild(item);
        });

    }catch(err){
        console.error('Branches load error:', err);
        listEl.innerHTML = `<div style="color:#b00">Error loading branches: ${err.message}</div>`;

        // Fallback dummy data so UI still shows map markers for review
        const fallback = [
            {id:1,name:'Colombo Central Pharmacy',lat:6.9271,lon:79.8612,address:'Colombo 01'},
            {id:2,name:'Kandy Health Center',lat:7.2906,lon:80.6337,address:'Kandy'},
            {id:3,name:'Galle Medical Hub',lat:6.0535,lon:80.2210,address:'Galle'},
            {id:4,name:'Jaffna Pharmacy',lat:9.6615,lon:80.0255,address:'Jaffna'}
        ];

        fallback.forEach(b => {
            const marker = L.marker([b.lat, b.lon]).addTo(map);
            marker.bindPopup(`<strong>${b.name}</strong><br>${b.address}`);

            const item = document.createElement('div');
            item.className = 'branch-item';
            item.innerHTML = `<strong>${b.name}</strong><div>${b.address}</div>`;
            item.addEventListener('click', () => {
                map.setView([b.lat, b.lon], 13);
                marker.openPopup();
            });
            listEl.appendChild(item);
        });
    }
})();
