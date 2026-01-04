#ifndef _LWIPOPTS_EXAMPLE_COMMONH_H
#define _LWIPOPTS_EXAMPLE_COMMONH_H


// Common settings used in most of the pico_w examples
// (see https://www.nongnu.org/lwip/2_1_x/group__lwip__opts.html for details)

// Pico SDK uses NO_SYS=1 even for the threadsafe background arch
// allow override in some examples
#ifndef NO_SYS
#define NO_SYS                      1
#endif
// allow override in some examples
#ifndef LWIP_SOCKET
#define LWIP_SOCKET                 0
#endif
#if PICO_CYW43_ARCH_POLL
#define MEM_LIBC_MALLOC             1
#else
// MEM_LIBC_MALLOC is incompatible with non polling versions
#define MEM_LIBC_MALLOC             0
#endif
#define MEM_ALIGNMENT               4
#ifndef MEM_SIZE
#define MEM_SIZE                    14000
#endif
#ifndef MEMP_NUM_SYS_TIMEOUT
#define MEMP_NUM_SYS_TIMEOUT        8
#endif
#ifndef LWIP_TIMERS
#define LWIP_TIMERS                 1
#endif
#ifndef SYS_LIGHTWEIGHT_PROT
#define SYS_LIGHTWEIGHT_PROT        1
#endif
#define MEMP_NUM_TCP_SEG            12
#define MEMP_NUM_ARP_QUEUE          2
#define PBUF_POOL_SIZE              8
#define PBUF_POOL_BUFSIZE           1024
#define LWIP_ARP                    1
#define LWIP_ETHERNET               1
#define LWIP_ICMP                   0
#define LWIP_RAW                    0
#define TCP_MSS                     536
#define TCP_WND                     (2 * TCP_MSS)
#define TCP_SND_BUF                 (2 * TCP_MSS)
#define TCP_SND_QUEUELEN            12
#define LWIP_DISABLE_TCP_SANITY_CHECKS 1
#define LWIP_NETIF_STATUS_CALLBACK  0
#define LWIP_NETIF_LINK_CALLBACK    0
#define LWIP_NETIF_HOSTNAME         0
#define LWIP_NETCONN                0
#define MEM_STATS                   0
#define SYS_STATS                   0
#define MEMP_STATS                  0
#define LINK_STATS                  0
// #define ETH_PAD_SIZE                2
#define LWIP_CHKSUM_ALGORITHM       3
#define LWIP_DHCP                   1
#define LWIP_IPV4                   1
#define LWIP_TCP                    1
#define LWIP_UDP                    1
#define LWIP_DNS                    0
#define LWIP_SNTP                   0
#define SNTP_SERVER_DNS             0
#define LWIP_TCP_KEEPALIVE          1
#define LWIP_NETIF_TX_SINGLE_PBUF   0
// Core locking not used with NO_SYS=1
#define DHCP_DOES_ARP_CHECK         0
#define LWIP_DHCP_DOES_ACD_CHECK    0

#define LWIP_HTTP_CLIENT            1
#define LWIP_HTTPD                  0
#define LWIP_ALTCP                  1
#define LWIP_ALTCP_TLS              0
#define LWIP_ALTCP_TLS_MBEDTLS      0


#define LWIP_DEBUG                  0
#define LWIP_STATS                  0
#define LWIP_STATS_DISPLAY          0

#endif /* __LWIPOPTS_H__ */
