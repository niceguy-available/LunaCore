const localtunnel = require('localtunnel');

(async () => {
  try {
    const tunnel = await localtunnel({ port: 3000, subdomain: `luna-isro-${Math.floor(1000 + Math.random() * 9000)}` });
    console.log(`LUNA_PUBLIC_URL=${tunnel.url}`);
    
    tunnel.on('close', () => {
      console.log('Tunnel closed');
    });

    tunnel.on('error', (err) => {
      console.error('Tunnel error:', err);
    });
  } catch (err) {
    console.error('Failed to create tunnel:', err);
  }
})();
